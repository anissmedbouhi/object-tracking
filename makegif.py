import imageio

# Generate list of filenames from 0001.jpg to 0511.jpg
filenames = [f"{i:04d}.jpg" for i in range(1, 512)]  # 1 to 511 inclusive

# Read each image and store in list
images = [imageio.imread(fname) for fname in filenames]

# Save as GIF (duration is in seconds per frame)
imageio.mimsave("output.gif", images, duration=0.05)  # 20 fps
