//
//  bilateral_filtering.metal
//  bilateral_grid
//
//  Created by Varun Malladi on 5/28/23.
//

#include <metal_stdlib>
using namespace metal;

kernel void horizontal_gaussian_blur(
    texture2d_array<half, access::read> grid [[texture(0)]],
    texture2d_array<half, access::write> output [[texture(1)]],
    constant float* kernelBuffer [[buffer(0)]],
    constant int* kernelSize [[buffer(1)]],
    uint3 index [[thread_position_in_grid]]
) {
    int halfway_index = ceil(float(kernelSize[0]-1) / 2 - 0.001);
    int radius = halfway_index;
    
    auto sum = half4(0);
    for (int i=-radius; i<=radius; ++i) {
        const uint offset = uint(halfway_index + i);
        const half weight = half(kernelBuffer[offset]);
        
        const uint modified_coord = max(0, int(index.x) + i);
        const uint2 coord_to_read = uint2(modified_coord, index.y);
        
        const half4 val = grid.read(coord_to_read, index.z);
        sum += weight * val;
    }
    
    output.write(sum, index.xy, index.z);
}

kernel void vertical_gaussian_blur(
    texture2d_array<half, access::read_write> output [[texture(1)]],
    constant float* kernelBuffer [[buffer(0)]],
    constant int* kernelSize [[buffer(1)]],
    uint3 index [[thread_position_in_grid]]
) {
    int halfway_index = ceil(float(kernelSize[0]-1) / 2 - 0.001);
    int radius = halfway_index;
    
    auto sum = half4(0);
    for (int i=-radius; i<=radius; ++i) {
        const uint offset = uint(halfway_index + i);
        const half weight = half(kernelBuffer[offset]);
        
        const uint modified_coord = max(0, int(index.y) + i);
        const uint2 coord_to_read = uint2(index.x, modified_coord);
        
        const half4 val = output.read(coord_to_read, index.z);
        sum += weight * val;
    }
    
    output.write(sum, index.xy, index.z);
}

kernel void depth_gaussian_blur(
    texture2d_array<half, access::read_write> output [[texture(1)]],
    constant float* kernelBuffer [[buffer(0)]],
    constant int* kernelSize [[buffer(1)]],
    uint3 index [[thread_position_in_grid]]
) {
    const int t = kernelSize[1];
    int halfway_index = ceil(float(t-1) / 2 - 0.001); // 2
    int radius = halfway_index; // 2
    
    auto sum = half4(0);
    for (int i=-radius; i<=radius; ++i) {
        const int offset = halfway_index + i;
        const uint weight_index = uint(kernelSize[0] + offset);
        const half weight = half(kernelBuffer[weight_index]);
        
        const uint modified_coord = clamp(
            int(index.z) + i,
            0,
            int(output.get_array_size()) - 1
        );
        const uint2 coord_to_read = uint2(index.x, index.y);
        
        const half4 val = output.read(coord_to_read, modified_coord);
        sum += weight * val;
    }
    
    output.write(sum, index.xy, index.z);
}
