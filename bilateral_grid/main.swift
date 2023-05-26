//
//  main.swift
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

import Foundation
import Metal
import MetalKit
import Cocoa

struct Vertex {
    var position: SIMD4<Float>
}

enum ComputeResourcesErrors: Error {
    case DefaultLibraryNotFound
    case ShaderFunctionNotFound
    case MakeCommandQueueFailed
    case ConstructPSOFailed
    case TextureCreationFailed
    case CommandBufferCreationFailed
    case CommandEncoderCreationFailed
    case SamplerCreationFailed
}

struct ComputeResources {
    let device: MTLDevice;
    let commandQueue: MTLCommandQueue;
    
    let replaceZeroPSO: MTLComputePipelineState;
    let constructBgPSO: MTLComputePipelineState;
    
    let slicePSO: MTLRenderPipelineState;
    let sliceKernelPSO: MTLComputePipelineState
    let bilinearSampler: MTLSamplerState
    
    
    let viewVertexData: [Vertex] = [
        Vertex(position: SIMD4<Float>(-1, -1, 0, 1)),
        Vertex(position: SIMD4<Float>(-1, 1, 0, 1)),
        Vertex(position: SIMD4<Float>(1, -1, 0, 1)),
        Vertex(position: SIMD4<Float>(1, 1, 0, 1))
    ];
    
    init(device_: MTLDevice) throws {
        device = device_;
        
        // Load the shader files with a .metal file extension in the project.
        guard let defaultLibrary = device.makeDefaultLibrary() else {
            throw ComputeResourcesErrors.DefaultLibraryNotFound;
        }
        
        // Create pipeline state objects.
        
        // Bilateral grid
        guard let replaceZeroFunction = defaultLibrary.makeFunction(name: "replace_with_zero") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        replaceZeroPSO = try device.makeComputePipelineState(function: replaceZeroFunction)
        
        guard let constructBgFunction = defaultLibrary.makeFunction(name: "construct_bilateral_grid") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound;
        }
        constructBgPSO = try device.makeComputePipelineState(function: constructBgFunction);
        
        // Slice
        guard let sliceVertexShader = defaultLibrary.makeFunction(name: "slice_vertex_shader") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound;
        }
        guard let sliceFragmentShader = defaultLibrary.makeFunction(name: "slice_fragment_shader") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound;
        }
        
        guard let sliceFunction = defaultLibrary.makeFunction(name: "slice_kernel") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        sliceKernelPSO = try device.makeComputePipelineState(function: sliceFunction)
        
        // Create sampler state.
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .nearest
        
        guard let bilinearSampler_ = device.makeSamplerState(descriptor: samplerDescriptor) else {
            throw ComputeResourcesErrors.SamplerCreationFailed
        }
        bilinearSampler = bilinearSampler_
        
        let renderDescriptor = MTLRenderPipelineDescriptor();
        renderDescriptor.vertexFunction = sliceVertexShader;
        renderDescriptor.fragmentFunction = sliceFragmentShader;
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm;
        
        slicePSO = try device.makeRenderPipelineState(descriptor: renderDescriptor);
        
        // Create command queue.
        guard let temp = device.makeCommandQueue() else {
            throw ComputeResourcesErrors.MakeCommandQueueFailed;
        }
        commandQueue = temp;
    }
    
    func loadImageAsTexture(imageURL: URL) throws -> MTLTexture {
        let loader = MTKTextureLoader(device: device)
        let loaderOptions: [MTKTextureLoader.Option : Any] = [
            .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue)
        ]
        
        let texture = try loader.newTexture(URL: imageURL, options: loaderOptions)
        
        print("loaded texture; pixel format: \(texture.pixelFormat)")
        
        return texture
    }
    
    func construct_bilateral_grid(image: MTLTexture) throws -> MTLTexture {
        // Hardcoded for testing.
        let depth = 1 // 15;
        
        // Create a texture for the bilateral grid.
        let textureDescriptor = MTLTextureDescriptor();
        textureDescriptor.textureType = .type3D;
        textureDescriptor.pixelFormat = .bgra8Unorm;
        textureDescriptor.width = image.width;
        textureDescriptor.height = image.height;
        textureDescriptor.depth = depth;
        textureDescriptor.usage = [.shaderRead, .shaderWrite];
        
        guard let grid_texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        grid_texture.label = "bilateral grid"
//        initializeTextureToZeros(texture: grid_texture);
        
        // Calculate grid and threadgroup size.
        let gridSize: MTLSize = MTLSizeMake(grid_texture.width, grid_texture.height, 1);
        
//        var threadGroupSize_: Int = constructBgPSO.maxTotalThreadsPerThreadgroup;
//        let num_texels = grid_texture.width * grid_texture.height;
//        if (threadGroupSize_ > num_texels) {
//            threadGroupSize_ = num_texels;
//        }
        let threadGroupSize: MTLSize = MTLSizeMake(16, 16, 1);
        
        // Start compute pass.
        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ComputeResourcesErrors.CommandBufferCreationFailed;
        }
        guard let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ComputeResourcesErrors.CommandEncoderCreationFailed;
        }
        
        // Zero initialize grid.
        computeEncoder.setComputePipelineState(replaceZeroPSO)
        computeEncoder.setTexture(grid_texture, index: 0)
        
        let zeroGridSize: MTLSize = MTLSizeMake(grid_texture.width, grid_texture.height, grid_texture.depth)
        
        computeEncoder.dispatchThreads(zeroGridSize, threadsPerThreadgroup: threadGroupSize)
        
        // Encode pipeline state object.
        computeEncoder.setComputePipelineState(constructBgPSO);
        computeEncoder.setTexture(image, index: 0);
        computeEncoder.setTexture(grid_texture, index: 1);

        // Encode compute command.
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        // End the compute pass.
        computeEncoder.endEncoding();
        
        // Execute and wait.
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        return grid_texture;
    }
    
    func slice(reference: MTLTexture, grid: MTLTexture) throws -> MTLTexture {
        // Set up render target.
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.width = reference.width
        textureDescriptor.height = reference.height
        textureDescriptor.usage = [.renderTarget, .shaderRead]
        guard let renderTexture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed
        }
        renderTexture.label = "slice render target"
        
        // Describe render pass.
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = renderTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0)
        
        // Start render pass.
        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ComputeResourcesErrors.CommandBufferCreationFailed
        }
        guard let renderEncoder: MTLRenderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw ComputeResourcesErrors.CommandEncoderCreationFailed
        }
        
        renderEncoder.setRenderPipelineState(slicePSO)
        renderEncoder.setVertexBytes(viewVertexData, length: MemoryLayout<Vertex>.stride * viewVertexData.count, index: 0)
        renderEncoder.setFragmentSamplerState(bilinearSampler, index: 0)
        
        // Draw.
        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: viewVertexData.count)
        
        // End render pass.
        renderEncoder.endEncoding()
        
        // Execute and wait.
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return renderTexture
    }
    
    func slice_ker(reference: MTLTexture, grid: MTLTexture) throws -> MTLTexture {
        // Create a texture for the sliced result.
        let textureDescriptor = MTLTextureDescriptor();
        textureDescriptor.textureType = .type2D;
        textureDescriptor.pixelFormat = .bgra8Unorm;
        textureDescriptor.width = reference.width;
        textureDescriptor.height = reference.height;
        textureDescriptor.usage = [.shaderRead, .shaderWrite];
        
        guard let result_texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        result_texture.label = "slice result"
        
        // Calculate grid and threadgroup size.
        let gridSize: MTLSize = MTLSizeMake(result_texture.width, result_texture.height, 1);
        let threadGroupSize: MTLSize = MTLSizeMake(16, 16, 1);
        
        // Start compute pass.
        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ComputeResourcesErrors.CommandBufferCreationFailed;
        }
        guard let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ComputeResourcesErrors.CommandEncoderCreationFailed;
        }

        // Encode pipeline state object.
        computeEncoder.setComputePipelineState(sliceKernelPSO)
        computeEncoder.setTexture(reference, index: 0)
        computeEncoder.setTexture(grid, index: 1)
        computeEncoder.setTexture(result_texture, index: 2)
        computeEncoder.setSamplerState(bilinearSampler, index: 0)

        // Encode compute command.
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        // End the compute pass.
        computeEncoder.endEncoding();
        
        // Execute and wait.
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        return result_texture;
    }
}

let device: MTLDevice = MTLCreateSystemDefaultDevice()!;

var cr = try! ComputeResources(device_: device);
let image_texture = try! cr.loadImageAsTexture(imageURL: URL(fileURLWithPath: "data/gile.jpg"));
print("loaded the image as a texture with dimensions (\(image_texture.width), \(image_texture.height), \(image_texture.depth))");
let bg = try! cr.construct_bilateral_grid(image: image_texture);
//let sliced = try! cr.slice(reference: image_texture, grid: bg)
let sliced = try! cr.slice_ker(reference: image_texture, grid: bg)

display_texture(device: device, texture: sliced)
