//
//  bilateral_grid.metal
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

#include <metal_stdlib>
using namespace metal;


kernel void replace_with_zero(
    texture2d_array<half, access::read_write> bilateralGrid [[texture(0)]],
    uint3 index [[thread_position_in_grid]]
) {
    bilateralGrid.write(half4(0.0), index.xy, index.z);
}

// Shadows CPU-side struct.
struct SamplingRates {
    float s_s;
    float s_t;
};

// Rec. 709 luma values for grayscale image conversion.
// From Apple sample code.
constant half3 kRec709Luma = half3(0.2126, 0.7152, 0.0722);

half rgb_to_grayscale(half3 rgb) {
    half unclamped = dot(rgb, kRec709Luma);
    return clamp(unclamped, half(0), half(1));
}

inline float3 pixel_to_grid_coordinate(
    uint2 pixel_index,
    half4 pixel_value,
    float s_s,
    float s_t
) {
    float x = float(pixel_index.x) / s_s;
    float y = float(pixel_index.y) / s_s;

    half gray_value = rgb_to_grayscale(pixel_value.rgb);
    float z = float(gray_value) / s_t;
    
    return float3(x, y, z);
}

/// Assumes `bilateralGrid` has been zero-initialized.
kernel void construct_bilateral_grid(
    texture2d<half, access::read> referenceImage [[texture(0)]],
    texture2d_array<half, access::read_write> bilateralGrid [[texture(1)]],
    constant SamplingRates& sampling_rates [[buffer(0)]],
//                                     threadgroup half4* running_sum [[threadgroup(0)]],
//                                     uint2 threadgroup_position [[thread_position_in_threadgroup]],
    uint2 index [[thread_position_in_grid]]
) {
    half4 reference_value = referenceImage.read(index);
    
    // Where to write in grid.
    auto unrounded = pixel_to_grid_coordinate(
        index,
        reference_value,
        1,
        sampling_rates.s_t
    );
    uint3 grid_index = uint3(
        round(unrounded.x),
        round(unrounded.y),
        round(unrounded.z)
    );

    // Update grid.
//    const uint2 threadgroup_size = uint2(sampling_rates.s_s, sampling_rates.s_s);
//
//    const uint TG_MEM_COUNT = 256;
//    threadgroup half4 running[TG_MEM_COUNT];
//    threadgroup half4 poss[TG_MEM_COUNT];
//
//    uint index_1d = threadgroup_size.x * threadgroup_position.x + threadgroup_position.y;
//    running[index_1d] = half4(reference_value.rgb, 1);
//    poss[index_1d] = half4(half(grid_index.x), half(grid_index.y), half(grid_index.z), 0);
//    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//    if (threadgroup_position.x == 0 && threadgroup_position.y == 0) {
//        const uint max_depth = 16;
//        half4 values[max_depth];
//
//        for (uint i=0; i<max_depth; ++i) {
//            values[i] = half4(0);
//        }
//
//        for (uint i=0; i<threadgroup_size.x*threadgroup_size.y; ++i) {
//            auto working_val = running[i];
//            auto working_pos = poss[i];
//
//
//
//            uint working_depth = working_pos.z;
//            values[working_depth] += working_val;
//        }
//
//        for (uint i=0; i<max_depth; ++i) {
//            half4 value = values[i];
//
//
//        }
//    }
    
    half4 previous_value = bilateralGrid.read(grid_index.xy, grid_index.z);
    half4 new_value = previous_value + half4(reference_value.rgb, 1);

    bilateralGrid.write(new_value, grid_index.xy, grid_index.z);
}

/// The grid size should correspond to the dimensions of the downsampled grid you want.
kernel void downsample_bilateral_grid(
    texture2d_array<half, access::read> original_grid [[texture(0)]],
    texture2d_array<half, access::write> new_grid [[texture(1)]],
    constant SamplingRates& sampling_rates [[buffer(0)]],
    uint3 index [[thread_position_in_grid]]
) {
    uint s_s = round(sampling_rates.s_s);
    
    uint x_index_start = s_s * index.x;
    uint y_index_start = s_s * index.y;

    int halfw = ceil(sampling_rates.s_s / 2 - 0.001);
    
    uint x_start = max(0, int(x_index_start) - halfw + 1);
    uint x_end = x_index_start + uint(halfw);
    uint y_start = max(0, int(y_index_start) - halfw + 1);
    uint y_end = y_index_start + uint(halfw);
    
    half4 running_sum = half4(0);
    for (uint x=x_start; x<x_end; ++x) {
        for (uint y=y_start; y<y_end; ++y) {
            running_sum += original_grid.read(uint2(x,y), index.z);
        }
    }
    new_grid.write(running_sum, index.xy, index.z);
}

kernel void slice_kernel(
    texture2d<half, access::read> reference [[texture(0)]],
    texture2d_array<half, access::sample> grid [[texture(1)]],
    texture2d<half, access::write> result [[texture(2)]],
    sampler grid_sampler [[sampler(0)]],
    constant SamplingRates& sampling_rates [[buffer(0)]],
    uint2 index [[thread_position_in_grid]]
) {
    auto grid_coord = pixel_to_grid_coordinate(
        index,
        reference.read(index),
        sampling_rates.s_s,
        sampling_rates.s_t
    );
    
    auto x = grid_coord.x;
    auto y = grid_coord.y;
    auto z = grid_coord.z;
    
    // Find the textures directly above/below the z.
    uint z_below = floor(z);
    uint z_above = ceil(z);
    
    // Get the bilinear interpolation values from those textures.
    half4 val_below = grid.sample(grid_sampler, float2(x, y), z_below);
    half4 val_above = grid.sample(grid_sampler, float2(x, y), z_above);
    
    // Interpolate these.
    half4 val = mix(
        val_below,
        val_above,
        z - z_below
    );
    
    // Convert from homogeneous coordinates.
    val = val / val.w; // max(val.w, half(1));
    val = clamp(val, half4(0), half4(1));
    val.w = 1;
    
//    half4 val_rounded = grid.sample(grid_sampler, float2(x, y), z);

    result.write(val, index);
}

kernel void gaussian_blur(
    texture2d_array<half, access::sample> grid [[texture(0)]],
    texture2d_array<half, access::write> output [[texture(1)]],
    sampler grid_sampler [[sampler(0)]],
    constant float* kernelBuffer [[buffer(0)]],
    constant int* kernelSizes [[buffer(1)]],
    uint3 index [[thread_position_in_grid]]
) {
    // Get the texture coordinate of the pixel we are working on.
    float2 texCoord = float2(index.xy); // / float2(grid.get_width(), grid.get_height());
    // ... and the fixed depth.
    float depth = float(index.z);
    
    int spatial_kernel_size = kernelSizes[0];
    float index_of_center = float(spatial_kernel_size + 1) / 2.0 - 1.0;
    int range_kernel_size = kernelSizes[1];
    float index_of_range_center = float(range_kernel_size + 1) / 2.0 - 1.0;
    
    half weight_sum = 0.0;
    half4 result = half4(0);
    
    // Spatial Gaussians
    for (int i=0; i<spatial_kernel_size; ++i) {
        // Weights are assumed to be isotropic spatially.
        float weight = kernelBuffer[i];
        weight_sum = weight_sum + weight;
        
        // Assuming the working pixel corresponds to the center of
        // the kernel, compute the relative offset from the center.
        float2 offset = float2(float(i) - index_of_center);
        // ... these correspond to pixel offsets, so normalize to get
        // texel offsets.
        //offset = offset / float2(grid.get_width(), grid.get_height());
        
        half4 x_color = grid.sample(grid_sampler, texCoord + offset.x, depth);
        half4 y_color = grid.sample(grid_sampler, texCoord + offset.y, depth);
        
        result += weight * (x_color + y_color);
    }
    
    // Range Gaussian
    for (int i=0; i<range_kernel_size; ++i) {
        float weight = kernelBuffer[spatial_kernel_size + i];
        weight_sum += weight;
        
        float offset = float(i) - index_of_range_center;
        float depth_to_sample = depth + offset;
        
        // Handle out of bounds.
        if (depth_to_sample < 0 || depth_to_sample > grid.get_array_size() - 1) {
            result += half4(0);
            continue;
        }
        
        half4 z_color = grid.sample(grid_sampler, texCoord, depth_to_sample);
        result += weight * z_color;
    }
    
//    result = result / weight_sum;
//    result = half4(result.xyz / result.w, 1);
    output.write(result, index.xy, index.z);
}
