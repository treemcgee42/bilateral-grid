//
//  main.swift
//  bilateral_grid
//
//  Created by Varun Malladi on 5/24/23.
//

import Foundation
import Metal

let device: MTLDevice = MTLCreateSystemDefaultDevice()!;

let s_s: Float = 3
let s_t: Float = 0.15
var cr = try! ComputeResources(device_: device, s_s: s_s, s_t: s_t);

let image_texture = try! loadImageAsTexture(device: device, imageURL: URL(fileURLWithPath: "data/rubiks_cube.png"));
print("loaded the image as a texture with dimensions (\(image_texture.width), \(image_texture.height), \(image_texture.depth))");

let bg = try! cr.construct_bilateral_grid(image: image_texture, referenceImage: image_texture);
print("constructed grid with dimensions \(bg.arrayLength) x (\(bg.width), \(bg.height))")


//var sliced: MTLTexture?
//sliced = try! cr.slice_ker(reference: image_texture, grid: bg)
//display_texture(device: device, texture: sliced!)
//try! saveTextureAsImage(sliced!, url: URL(filePath: "unfiltered.png"))
//sliced = nil


var bilateral_filter_result: MTLTexture?
bilateral_filter_result = try! cr.bilateral_filtering(reference: image_texture, grid: bg, spatialSigma: s_s, rangeSigma: 3, spatialKernelSize: 5)
display_texture(device: device, texture: bilateral_filter_result!)
//try! saveTextureAsImage(bilateral_filter_result!, url: URL(filePath: "filtered.png"))
//bilateral_filter_result = nil
