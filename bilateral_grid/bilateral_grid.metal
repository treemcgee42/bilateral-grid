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
constant half3 kRec709Luma = half3(0.2989, 0.587, 0.114); // half3(0.2126, 0.7152, 0.0722);

half rgb_to_grayscale(half3 rgb) {
    half unclamped = dot(rgb, kRec709Luma);
    return unclamped; // clamp(unclamped, half(0), half(1));
}

inline float3 pixel_to_grid_coordinate(
    uint2 pixel_index,
    half4 pixel_value,
    float s_s,
    float s_t
) {
    const float x = float(pixel_index.x) / s_s;
    const float y = float(pixel_index.y) / s_s;

    const half gray_value = rgb_to_grayscale(pixel_value.rgb);
    const float z = float(gray_value) / s_t;
    
    return float3(x, y, z);
}

half calculate_scaling(float s_s) {
    return s_s * s_s;
}

/// Assumes `bilateralGrid` has been zero-initialized.
kernel void construct_bilateral_grid(
    texture2d<half, access::read> image [[texture(0)]],
    texture2d<half, access::read> reference_image [[texture(1)]],
    texture2d_array<half, access::read_write> bilateralGrid [[texture(2)]],
    constant SamplingRates& sampling_rates [[buffer(0)]],
    uint2 index [[thread_position_in_grid]]
) {
    half4 image_value = image.read(index);
    half4 reference_value = reference_image.read(index);
    
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
    
    half4 previous_value = bilateralGrid.read(grid_index.xy, grid_index.z);
    half4 new_value = previous_value + half4(image_value.rgb, 1);

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

    float halfway = sampling_rates.s_s / 2;
    
    uint x_start = max(uint(0), uint(ceil(float(x_index_start) - halfway - 0.1)));
    uint x_end = min(original_grid.get_width(), uint(ceil(float(x_index_start) + halfway - 0.1)));
    uint y_start = max(uint(0), uint(ceil(float(y_index_start) - halfway - 0.1)));
    uint y_end = min(original_grid.get_height(), uint(ceil(float(y_index_start) + halfway - 0.1)));
    
    half4 running_sum = half4(0, 0, 0, 0);
    for (uint x=x_start; x<x_end; ++x) {
        for (uint y=y_start; y<y_end; ++y) {
            half4 val = original_grid.read(uint2(x,y), index.z);
            val /= calculate_scaling(sampling_rates.s_s);
            running_sum += val;
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
        half(z) - half(z_below)
    );
    
    // Convert from homogeneous coordinates.
    if (val.w > 0.001) {
        val /= val.w; // max(val.w, half(1));
        val = clamp(val, half4(0), half4(1));
        val.w = 1;
    }

    result.write(val, index);
}
