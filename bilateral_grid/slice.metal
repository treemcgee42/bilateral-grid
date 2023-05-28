//
//  slice.metal
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

#include <metal_stdlib>
using namespace metal;

// Shadows CPU-side struct.
struct SamplingRates {
    float s_s;
    float s_t;
};

// Rec. 709 luma values for grayscale image conversion.
// From Apple sample code.
constant half3 kRec709Luma = half3(0.2126, 0.7152, 0.0722);







// Shadows CPU-side struct (the type of the elements in the vertex buffer).
struct Vertex {
    float4 position [[position]];
};

// Just passes through the vertex positions (expected to correspond to the four
// corners, already in z=0 clip space.
vertex Vertex slice_vertex_shader(
    const device Vertex* vertices [[buffer(0)]],
    uint vertexID [[vertex_id]]
) {
    return vertices[vertexID];
}



fragment half4 slice_fragment_shader(
    Vertex v [[stage_in]],
    texture2d<half, access::read> reference [[texture(0)]],
    texture3d<half, access::sample> grid [[texture(1)]],
    sampler grid_sampler [[sampler(0)]]
) {
    // Hardcoded for testing.
    float s_s = 1; // 16;
    float s_t = 1; // 0.07;
    
    
    // Get the value of the reference image at the specified point.
    // The position coordinates passed to the fragment shader are
    // clip-space, ranging from -1 to 1 on both axes.
    
    float x = v.position.x / s_s;
    float y = v.position.y / s_s;
    float z = dot(reference.read(uint2(round(1600 * v.position.x), round(1067 * v.position.y))).rgb, kRec709Luma) / s_t;
    
    // Find the textures directly above/below the z.
    uint z_below = floor(z);
    uint z_above = ceil(z);
    
    // Get the bilinear interpolation values from those textures.
    half4 val_below = grid.sample(grid_sampler, float3(x, y, z_below));
    half homogeneized_val_below = val_below.x / val_below.y;
    val_below = half4(homogeneized_val_below);
    val_below.w = 1;
    
    half4 val_above = grid.sample(grid_sampler, float3(x, y, z_above));
    half homogeneized_val_above = val_above.x / val_above.y;
    val_above = half4(homogeneized_val_above);
    val_above.w = 1;
    
    // Interpolate these.
    half4 val = mix(val_below, val_above, (z-float(z_below))/float(z_above - z_below));
    
    return val;
}
