//
//  renderer.metal
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
    float2 tex_coord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 tex_coord;
};

vertex VertexOut basic_vertex_shader(
    const device VertexIn* vertices [[buffer(0)]],
    uint index [[vertex_id]]
) {
    VertexOut to_return;
    to_return.position = float4(vertices[index].position, 0.0, 1.0);
    to_return.tex_coord = vertices[index].tex_coord;
    
    return to_return;
}

fragment float4 basic_fragment_shader(
    VertexOut in [[stage_in]],
    texture2d<float, access::sample> texture [[texture(0)]],
    sampler sampler [[sampler(0)]]
) {
    float4 color = texture.sample(sampler, in.tex_coord);
    return color;
}
