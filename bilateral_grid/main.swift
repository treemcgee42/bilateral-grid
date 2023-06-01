//
//  main.swift
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

import Foundation
import Metal

func testSlicing(imagePath: String, s_s: Float, s_t: Float) throws {
    let imageTexture = try loadImageAsTexture(device: device, imageURL: URL(fileURLWithPath: imagePath))
    
    let bg = try cr.construct_bilateral_grid(image: imageTexture, referenceImage: imageTexture)
    let output = try cr.slice_ker(reference: imageTexture, grid: bg)
    
    try saveTextureAsImage(output, url: URL(filePath: "results/gile.jpg"))
}

func testBilateralFiltering(imagePath: String, computeResources: ComputeResources, s_s: Float, s_t: Float) throws {
    let image_texture = try! loadImageAsTexture(device: device, imageURL: URL(fileURLWithPath: imagePath));
    
    let startTime = DispatchTime.now()
    
    let bg = try! cr.construct_bilateral_grid(image: image_texture, referenceImage: image_texture)
    
    let bilateral_filter_result = try cr.bilateral_filtering(reference: image_texture, grid: bg, spatialSigma: s_s, rangeSigma: 3, spatialKernelSize: 5)
    
    let endTime = DispatchTime.now()
    let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
    let elapsedTime = Double(nanoTime) / 1_000_000_000
    print("time elapsed: \(elapsedTime)")
    
    try saveTextureAsImage(bilateral_filter_result, url: URL(filePath: "results/rubiks_\(s_s)_\(s_t).jpg"))
//    display_texture(device: device, texture: bilateral_filter_result)
}

func testCrossBilateralFiltering(imagePath: String, edgeImagePath: String, computeResources: ComputeResources, s_s: Float, s_t: Float) throws {
    let image_texture = try loadImageAsTexture(device: device, imageURL: URL(fileURLWithPath: imagePath));
    let edge_image_texture = try loadImageAsTexture(device: device, imageURL: URL(fileURLWithPath: edgeImagePath));
    
    let result = try computeResources.cross_bilateral_filtering(image: image_texture, edge_image: edge_image_texture, spatialSigma: s_s, rangeSigma: s_t)
    
//    try saveTextureAsImage(result, url: URL(filePath: "results/cake_\(s_s)_\(s_t).png"))
    display_texture(device: computeResources.device, texture: result)
}

let device: MTLDevice = MTLCreateSystemDefaultDevice()!;
var cr = try! ComputeResources(device_: device, s_s: s_s, s_t: s_t);

//let s_s: Float = 6
//let s_t: Float = 0.6
//try! testSlicing(imagePath: "data/gile.jpg", s_s: s_s, s_t: s_t)

//let s_s: Float = 4
//let s_t: Float = 0.8
//try! testBilateralFiltering(imagePath: "data/rubiks_cube.png", computeResources: cr, s_s: s_s, s_t: s_t)

let s_s: Float = 1
let s_t: Float = 0.5
try! testCrossBilateralFiltering(imagePath: "data/teapot_flash.jpg", edgeImagePath: "data/teapot_no_flash.jpg", computeResources: cr, s_s: s_s, s_t: s_t)
