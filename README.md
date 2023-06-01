#  Bilateral grid

The meat of this project lies the CPU file `bilateral_grid.swift` and the shaders it calls, which are `bilateral_grid.metal`, `slice.metal`, and `bilateral_filtering.metal`. The naming is hopefully self-explanatory.

In `main.swift` you will find a few testing functions, as well as where those are actually called. 

The remaining files are helpers. `renderer.swift` and `renderer.metal` are responsible for displaying the results of grid slicing in a window on the system. `io.swift` is responsible for loading images into textures and saving textures as images.
