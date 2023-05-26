//
//  renderer.swift
//  bilateral_grid
//
//  Created by Varun Malladi on 5/25/23.
//

import Foundation
import Metal
import MetalKit

private struct VertexT {
    var position: SIMD2<Float>
    var texel: SIMD2<Float>
}

class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    var texture: MTLTexture
    let sampler: MTLSamplerState
    var pipelineState: MTLRenderPipelineState
    
    private let viewVertexData: [VertexT] = [
        // bottom left
        VertexT(position: SIMD2<Float>(-1, -1), texel: SIMD2<Float>(0, 1)),
        // bottom right
        VertexT(position: SIMD2<Float>(1, -1), texel: SIMD2<Float>(1, 1)),
        // top left
        VertexT(position: SIMD2<Float>(-1, 1), texel: SIMD2<Float>(0, 0)),
        
        // top left
        VertexT(position: SIMD2<Float>(-1, 1), texel: SIMD2<Float>(0, 0)),
        // bottom right
        VertexT(position: SIMD2<Float>(1, -1), texel: SIMD2<Float>(1, 1)),
        // top right
        VertexT(position: SIMD2<Float>(1, 1), texel: SIMD2<Float>(1, 0))
    ];
    
    init(device_: MTLDevice, texture_: MTLTexture) {
        device = device_
        texture = texture_
        
        commandQueue = device.makeCommandQueue()!
        
        // Get vertex and fragment functions.
        let defaultLibrary = device.makeDefaultLibrary()!
        
        let vertexFunc = defaultLibrary.makeFunction(name: "basic_vertex_shader")
        let fragmentFunc = defaultLibrary.makeFunction(name: "basic_fragment_shader")
        
        // Create render pipeline.
        let renderDescriptor = MTLRenderPipelineDescriptor();
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm;
        renderDescriptor.vertexFunction = vertexFunc
        renderDescriptor.fragmentFunction = fragmentFunc
        
        pipelineState = try! device.makeRenderPipelineState(descriptor: renderDescriptor)
        
        // Create sampler.
        let samplerDescriptor = MTLSamplerDescriptor()
        sampler = device.makeSamplerState(descriptor: samplerDescriptor)!
    }
    
    func draw(in view: MTKView) {
        // Create a command buffer.
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        
        // Describe render pass.
        guard let renderPassDescriptor = view.currentRenderPassDescriptor else {
            return
        }
        
        // Create command encoder.
        let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        
        // Encode.
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setFragmentTexture(texture, index: 0)
        renderEncoder.setFragmentSamplerState(sampler, index: 0)
        renderEncoder.setVertexBytes(viewVertexData, length: MemoryLayout<VertexT>.stride * viewVertexData.count, index: 0)
        renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: viewVertexData.count)
        
        renderEncoder.endEncoding()
        
        // Execute.
        guard let drawable = view.currentDrawable else {
            return
        }
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        return
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var metalView: MTKView!
    
    init(metalView: MTKView) {
        self.metalView = metalView
    }
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Create the window.
        let windowRect = NSRect(x: 0, y: 0, width: metalView.drawableSize.width, height: metalView.drawableSize.height)
        window = NSWindow(contentRect: windowRect, styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        window.title = "CS73 Final Project"
        
        // Add the MTKView to the window's content view.
        window.contentView?.addSubview(metalView)
        
        // Make the window visible.
        window.makeKeyAndOrderFront(nil)
    }
}

func display_texture(device: MTLDevice, texture: MTLTexture) {
    let renderer = Renderer(device_: device, texture_: texture)
    let metalView = MTKView(frame: CGRect(x: 0, y: 0, width: renderer.texture.width, height: renderer.texture.height), device: device)
    metalView.colorPixelFormat = MTLPixelFormat.bgra8Unorm
    metalView.delegate = renderer

    let appDelegate = AppDelegate(metalView: metalView)
    NSApplication.shared.delegate = appDelegate
    NSApplication.shared.run()
}
