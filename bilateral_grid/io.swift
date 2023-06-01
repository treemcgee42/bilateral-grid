//
//  io.swift
//  bilateral_grid
//
//  Functions for loading images to texture and vice-versa.
//
//  Created by Varun Malladi on 5/28/23.
//
//  Credits:
//  - https://stackoverflow.com/questions/52920497/swift-metal-save-bgra8unorm-texture-to-png-file
//

import Foundation
import Metal
import MetalKit
import Accelerate
import UniformTypeIdentifiers

enum TextureIOError: Error {
    case LoadingImageAsTextureFailed
    case InvalidImageDestination
}

func loadImageAsTexture(device: MTLDevice, imageURL: URL) throws -> MTLTexture {
    let loader = MTKTextureLoader(device: device)
    let loaderOptions: [MTKTextureLoader.Option : Any] = [
        .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
        .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue)
    ]
    
    let texture = try loader.newTexture(URL: imageURL, options: loaderOptions)
    
    return texture
}

func saveTextureAsImage(_ texture: MTLTexture, url: URL) throws {
    guard let image = loadTextureAsImage(texture: texture) else {
        throw TextureIOError.LoadingImageAsTextureFailed
    }
    
    guard let imageDestination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw TextureIOError.InvalidImageDestination
    }
    
    CGImageDestinationAddImage(imageDestination, image, nil)
    CGImageDestinationFinalize(imageDestination)
}

private func loadTextureAsImage(texture: MTLTexture) -> CGImage? {
    assert(texture.pixelFormat == .bgra8Unorm)
    
    // Create a CPU-accessible buffer for the texture data.
    let bytesPerPixel = 4 * MemoryLayout<UInt8>.size
    let bytesPerRow = texture.width * bytesPerPixel
    let bytesPerImage = bytesPerRow * texture.height
    
    let imageBuffer = UnsafeMutableRawPointer.allocate(byteCount: bytesPerImage, alignment: bytesPerPixel)
    defer { imageBuffer.deallocate() }
    
    // Copy the data over.
    texture.getBytes(
        imageBuffer,
        bytesPerRow: bytesPerRow,
        bytesPerImage: bytesPerImage,
        from: MTLRegionMake2D(0, 0, texture.width, texture.height),
        mipmapLevel: 0,
        slice: 0
    )
    
    swizzleBGRA8toRGBA8(imageBuffer, width: texture.width, height: texture.height)
    
    // Describe the image.
    guard let colorSpace = CGColorSpace(name: CGColorSpace.linearSRGB) else {
        return nil
    }
    let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue
    guard let bitmapContext = CGContext(data: nil, width: texture.width, height: texture.height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
        return nil
    }
    
    // Transfer data to image.
    bitmapContext.data?.copyMemory(from: imageBuffer, byteCount: bytesPerImage)
    
    // Make image.
    let image = bitmapContext.makeImage()
    return image
}

private func swizzleBGRA8toRGBA8(_ bytes: UnsafeMutableRawPointer, width: Int, height: Int) {
    let vWidth = vImagePixelCount(width)
    let vHeight = vImagePixelCount(height)
    let bytesPerRow = width * 4 // * MemoryLayout<UInt8>.size
    
    var souceBuffer = vImage_Buffer(data: bytes, height: vHeight, width: vWidth, rowBytes: bytesPerRow)
    var destBuffer = vImage_Buffer(data: bytes, height: vHeight, width: vWidth, rowBytes: bytesPerRow)
    
    // Describes how to rearrange the indices of the BGRA representation
    // to get an RGBA representation.
    var swizzleMask: [UInt8] = [2, 1, 0, 3]
    
    vImagePermuteChannels_ARGB8888(&souceBuffer, &destBuffer, &swizzleMask, vImage_Flags(kvImageNoFlags))
}
