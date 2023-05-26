//
//  bilateral_grid.metal
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

#include <metal_stdlib>
using namespace metal;

kernel void replace_with_zero(texture3d<half, access::read_write> bilateralGrid [[texture(0)]], uint3 index [[thread_position_in_grid]]) {
    bilateralGrid.write(half4(0.0), index);
}

// Rec. 709 luma values for grayscale image conversion.
// From Apple sample code.
constant half3 kRec709Luma = half3(0.2126, 0.7152, 0.0722);

/// Assumes `bilateralGrid` has been zero-initialized.
kernel void construct_bilateral_grid(
    texture2d<half, access::read> referenceImage [[texture(0)]],
    texture3d<half, access::read_write> bilateralGrid [[texture(1)]],
    uint2 index [[thread_position_in_grid]]
) {
    // Hardcoded for testing.
    half s_s = 1; // 16;
    half s_t = 1; // 0.07;

    // Get grayscale value at index.
    half4 reference_value = referenceImage.read(index);
    half gray_value = dot(reference_value.rgb, kRec709Luma);

    // Compute values to add to grid.
    uint3 grid_index = uint3(
        round(index.x / s_s),
        round(index.y / s_s),
        round(gray_value / s_t)
    );
    half2 to_add = half2(gray_value, 1);

    // Update grid.
    half4 previous_value = bilateralGrid.read(grid_index);
    half4 new_value = previous_value + half4(to_add, 0, 1);
    
    new_value = half4(new_value.x/new_value.y);
    new_value.w = 1;
    bilateralGrid.write(new_value, grid_index);
}
