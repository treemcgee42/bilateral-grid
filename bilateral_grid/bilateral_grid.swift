//
//  bilateral_grid.swift
//  bilateral_grid
//
//  Created by Varun Malladi on 5/28/23.
//

import Foundation
import Metal

struct Vertex {
    var position: SIMD4<Float>
}

struct SamplingRates {
    var s_s: Float
    var s_t: Float
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
    case BufferCreationFailed
}

struct ComputeResources {
    let device: MTLDevice;
    let commandQueue: MTLCommandQueue;
    
    let replaceZeroPSO: MTLComputePipelineState;
    let constructBgPSO: MTLComputePipelineState;
    let downsampleBgPSO: MTLComputePipelineState
    
    let slicePSO: MTLRenderPipelineState;
    let horizontalGaussianBlurPSO: MTLComputePipelineState
    let verticalGaussianBlurPSO: MTLComputePipelineState
    let depthGaussianBlurPSO: MTLComputePipelineState
    let sliceKernelPSO: MTLComputePipelineState
    let bilinearSampler: MTLSamplerState
    
    let viewVertexData: [Vertex] = [
        Vertex(position: SIMD4<Float>(-1, -1, 0, 1)),
        Vertex(position: SIMD4<Float>(-1, 1, 0, 1)),
        Vertex(position: SIMD4<Float>(1, -1, 0, 1)),
        Vertex(position: SIMD4<Float>(1, 1, 0, 1))
    ];
    
    let threadGroupData = SIMD4<Float>(0, 0, 0, 0)
    
    let s_s: Float
    let s_t: Float
    let samplingRatesBuffer: MTLBuffer
    
    init(device_: MTLDevice, s_s: Float, s_t: Float) throws {
        device = device_;
        
        // Create uniform buffer to hold sampling rates.
        self.s_s = s_s
        self.s_t = s_t
        var rates = SamplingRates(s_s: s_s, s_t: s_t)
        guard let buffer = device.makeBuffer(bytes: &rates, length: MemoryLayout<SamplingRates>.stride) else {
            throw ComputeResourcesErrors.BufferCreationFailed
        }
        samplingRatesBuffer = buffer
        
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
        
        guard let downsampleBgFunction = defaultLibrary.makeFunction(name: "downsample_bilateral_grid") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        downsampleBgPSO = try device.makeComputePipelineState(function: downsampleBgFunction)
        
        // Guassian blur
        guard let horizontalGuassianBlurFunction = defaultLibrary.makeFunction(name: "horizontal_gaussian_blur") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        horizontalGaussianBlurPSO = try device.makeComputePipelineState(function: horizontalGuassianBlurFunction)
        
        guard let verticalGuassianBlurFunction = defaultLibrary.makeFunction(name: "vertical_gaussian_blur") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        verticalGaussianBlurPSO = try device.makeComputePipelineState(function: verticalGuassianBlurFunction)
        
        guard let depthGuassianBlurFunction = defaultLibrary.makeFunction(name: "depth_gaussian_blur") else {
            throw ComputeResourcesErrors.ShaderFunctionNotFound
        }
        depthGaussianBlurPSO = try device.makeComputePipelineState(function: depthGuassianBlurFunction)
        
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
        samplerDescriptor.normalizedCoordinates = false
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .notMipmapped
        samplerDescriptor.sAddressMode = .clampToZero
        samplerDescriptor.tAddressMode = .clampToZero
        samplerDescriptor.maxAnisotropy = 1
        
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
    
    func construct_bilateral_grid(image: MTLTexture, referenceImage: MTLTexture) throws -> MTLTexture {
        // Represents how many slices are in the grid.
        // round(1 / s_t) represents the highest level of the grid,
        // and we add 1 because level 0 can also be occupied.
        let depth = Int(round(1.0 / s_t)) + 1
        
        // Create a texture for the bilateral grid.
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.pixelFormat = image.pixelFormat
        textureDescriptor.width = image.width
        textureDescriptor.height = image.height
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.arrayLength = depth
        
        guard let grid_texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        grid_texture.label = "bilateral grid"
        
        let textureDescriptor2 = MTLTextureDescriptor()
        textureDescriptor2.textureType = .type2DArray
        textureDescriptor2.pixelFormat = image.pixelFormat
        textureDescriptor2.width = Int(round(Float(image.width-1) / s_s)) + 1
        textureDescriptor2.height = Int(round(Float(image.height-1) / s_s)) + 1
        textureDescriptor2.usage = [.shaderRead, .shaderWrite]
        textureDescriptor2.arrayLength = depth
        
        guard let grid_texture2 = device.makeTexture(descriptor: textureDescriptor2) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        grid_texture2.label = "ds bilateral grid"
        
        // Calculate threadgroup size.
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
        
        let zeroGridSize: MTLSize = MTLSizeMake(grid_texture.width, grid_texture.height, grid_texture.arrayLength)
        
        computeEncoder.dispatchThreads(zeroGridSize, threadsPerThreadgroup: threadGroupSize)
        
        // Construct bilateral grid.
        computeEncoder.setComputePipelineState(constructBgPSO);
        computeEncoder.setTexture(image, index: 0);
        computeEncoder.setTexture(referenceImage, index: 1);
        computeEncoder.setTexture(grid_texture, index: 2)
        computeEncoder.setBuffer(samplingRatesBuffer, offset: 0, index: 0)
        
        let bgGridSize: MTLSize = MTLSizeMake(image.width, image.height, 1)

        computeEncoder.dispatchThreads(bgGridSize, threadsPerThreadgroup: threadGroupSize);
        
        // Downsample
        computeEncoder.setComputePipelineState(downsampleBgPSO)
        computeEncoder.setTexture(grid_texture, index: 0)
        computeEncoder.setTexture(grid_texture2, index: 1)
        computeEncoder.setBuffer(samplingRatesBuffer, offset: 0, index: 0)
        
        let dsGridSize: MTLSize = MTLSizeMake(grid_texture2.width, grid_texture2.height, grid_texture2.arrayLength)
        
        computeEncoder.dispatchThreads(dsGridSize, threadsPerThreadgroup: threadGroupSize)
        
        // End the compute pass.
        computeEncoder.endEncoding();
        
        // Execute and wait.
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        return grid_texture2;
    }
    
    func slice_ker(reference: MTLTexture, grid: MTLTexture) throws -> MTLTexture {
        // Create a texture for the sliced result.
        let textureDescriptor = MTLTextureDescriptor();
        textureDescriptor.textureType = .type2D;
        textureDescriptor.pixelFormat = reference.pixelFormat;
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
        computeEncoder.setBuffer(samplingRatesBuffer, offset: 0, index: 0)

        // Encode compute command.
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        // End the compute pass.
        computeEncoder.endEncoding();
        
        // Execute and wait.
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        return result_texture;
    }
    
    func computeGaussianKernel(sigma: Float, kernelSize: Int) -> [Float] {
        var kernel = [Float](repeating: 0.0, count: kernelSize)
        var weight_sum: Float = 0.0
        
        let twoSigmaSquared = 2 * sigma * sigma
        let coefficient = 1 / (sigma * sqrt(2 * .pi))
        
        let bound = kernelSize / 2
        for x in -bound...bound {
            let exponent = -(Float(x*x) / twoSigmaSquared)
            let weight = coefficient * exp(exponent)
            
            kernel[x+bound] = weight
            weight_sum += weight
        }
        
        // Normalize.
        for i in 0..<kernelSize {
            kernel[i] = kernel[i] / weight_sum
        }
        
        return kernel
    }
    
    func bilateral_filtering(reference: MTLTexture, grid: MTLTexture, spatialSigma: Float, rangeSigma: Float, spatialKernelSize: Int = 5, rangeKernelSize: Int = 5) throws -> MTLTexture {
        // Compute Gaussian kernels.
        let spatialKernel: [Float] = computeGaussianKernel(sigma: spatialSigma, kernelSize: spatialKernelSize)
        let rangeKernel: [Float] = computeGaussianKernel(sigma: rangeSigma, kernelSize: rangeKernelSize)
        
        var kernels = spatialKernel + rangeKernel
        var kernelSizes: [Int32] = [Int32(spatialKernelSize), Int32(rangeKernelSize)]
        
        // Create texture for blurred grid.
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.width = grid.width
        textureDescriptor.height = grid.height
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.arrayLength = grid.arrayLength
        
        guard let grid_texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        grid_texture.label = "temp filter grid"
        
        // Calculate threadgroup size.
        let threadGroupSize: MTLSize = MTLSizeMake(16, 16, 1);
        
        // Start compute pass.
        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ComputeResourcesErrors.CommandBufferCreationFailed;
        }
        guard let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ComputeResourcesErrors.CommandEncoderCreationFailed;
        }
        
        // Construct blurred grid.
        computeEncoder.setTexture(grid, index: 0);
        computeEncoder.setTexture(grid_texture, index: 1);
        
        guard let kernelBuffer = device.makeBuffer(bytes: &kernels, length: kernels.count * MemoryLayout<Float>.stride) else {
            throw ComputeResourcesErrors.BufferCreationFailed
        }
        computeEncoder.setBuffer(kernelBuffer, offset: 0, index: 0)
        guard let kernelSizeBuffer = device.makeBuffer(bytes: &kernelSizes, length: kernelSizes.count * MemoryLayout<Int32>.stride) else {
            throw ComputeResourcesErrors.BufferCreationFailed
        }
        computeEncoder.setBuffer(kernelSizeBuffer, offset: 0, index: 1)
        
        let gridSize: MTLSize = MTLSizeMake(grid.width, grid.height, grid.arrayLength)

        computeEncoder.setComputePipelineState(horizontalGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        computeEncoder.setComputePipelineState(verticalGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        computeEncoder.setComputePipelineState(depthGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);

        computeEncoder.endEncoding();
        
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        // Slice.
        let sliced = try slice_ker(reference: reference, grid: grid_texture)
        
        return sliced
    }
    
    func cross_bilateral_filtering(
        image: MTLTexture,
        edge_image: MTLTexture,
        spatialSigma: Float,
        rangeSigma: Float,
        spatialKernelSize: Int = 5,
        rangeKernelSize: Int = 5
    ) throws -> MTLTexture {
        // Construct grid.
        let grid = try construct_bilateral_grid(image: image, referenceImage: edge_image)
        
        // Compute Gaussian kernels.
        let spatialKernel: [Float] = computeGaussianKernel(sigma: spatialSigma, kernelSize: spatialKernelSize)
        let rangeKernel: [Float] = computeGaussianKernel(sigma: rangeSigma, kernelSize: rangeKernelSize)
        
        var kernels = spatialKernel + rangeKernel
        var kernelSizes: [Int32] = [Int32(spatialKernelSize), Int32(rangeKernelSize)]
        
        // Create texture for blurred grid.
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type2DArray
        textureDescriptor.pixelFormat = .bgra8Unorm
        textureDescriptor.width = grid.width
        textureDescriptor.height = grid.height
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.arrayLength = grid.arrayLength
        
        guard let grid_texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw ComputeResourcesErrors.TextureCreationFailed;
        }
        grid_texture.label = "temp filter grid"
        
        // Calculate threadgroup size.
        let threadGroupSize: MTLSize = MTLSizeMake(16, 16, 1);
        
        // Start compute pass.
        guard let commandBuffer: MTLCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw ComputeResourcesErrors.CommandBufferCreationFailed;
        }
        guard let computeEncoder: MTLComputeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ComputeResourcesErrors.CommandEncoderCreationFailed;
        }
        
        // Construct blurred grid.
        computeEncoder.setTexture(grid, index: 0);
        computeEncoder.setTexture(grid_texture, index: 1);
        
        guard let kernelBuffer = device.makeBuffer(bytes: &kernels, length: kernels.count * MemoryLayout<Float>.stride) else {
            throw ComputeResourcesErrors.BufferCreationFailed
        }
        computeEncoder.setBuffer(kernelBuffer, offset: 0, index: 0)
        guard let kernelSizeBuffer = device.makeBuffer(bytes: &kernelSizes, length: kernelSizes.count * MemoryLayout<Int32>.stride) else {
            throw ComputeResourcesErrors.BufferCreationFailed
        }
        computeEncoder.setBuffer(kernelSizeBuffer, offset: 0, index: 1)
        
        let gridSize: MTLSize = MTLSizeMake(grid.width, grid.height, grid.arrayLength)

        computeEncoder.setComputePipelineState(horizontalGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        computeEncoder.setComputePipelineState(verticalGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);
        
        computeEncoder.setComputePipelineState(depthGaussianBlurPSO);
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroupSize);

        computeEncoder.endEncoding();
        
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
        
        // Slice.
        let sliced = try slice_ker(reference: edge_image, grid: grid_texture)
        
        return sliced
    }
}
