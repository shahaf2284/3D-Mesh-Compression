
import os
from PIL import Image
#from adodbapi.ado_consts import directions
import numpy as np
import gzip
import shutil
import zlib
import bz2
import lzma
import cv2
from collections import deque


def ipgz_Compression(data):
    compressed_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\gzip"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    compressed_size = os.path.getsize(compressed_file_path)
    print("=============================================")
    print("For gzip type compression")
    print("Image size after compression:", compressed_size, "byte")


def zlib_Compression(data):
    compressed_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\zlib"
    compressed_data = zlib.compress(data)  # Compress the data using zlib.compress
    with open(compressed_file_path, 'wb') as f_out:
        f_out.write(compressed_data)  # Write the compressed data to the file

    # Measure the size of the compressed image
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For zlib type compression")
    print("Compressed size:", compressed_size, "bytes")





def BZIP2_Compression(data):
    compressed_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\BZIP2"
    with open(compressed_file_path, 'wb') as f_out:
        compressed_data = bz2.compress(data)
        f_out.write(compressed_data)
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For BZIP2 type compression")
    print("Compressed size:", compressed_size, "bytes")


def LP_transformation(image_path, data):
    compressed_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\LP"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    # Apply Laplacian transformation
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Convert the Laplacian result back to uint8
    laplacian_uint8 = np.uint8(np.absolute(laplacian))
    # Save the Laplacian-transformed image
    output_image_path = r"C:\Users\97252\Desktop\Copressrion_examples\laplacian_transformed_image.jpg"
    cv2.imwrite(output_image_path, laplacian_uint8)
    # Get the size of the compressed image
    # compressed_size = os.path.getsize(output_image_path)
    print("=============================================")
    print("For Laplacian Transformation type compression")
    print("Compressed size:",len(output_image_path) , "bytes")
    print(laplacian_uint8)


def Quantized(image_path):
    image = cv2.imread(image_path)
    # Convert the image to a format suitable for quantization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = np.float32(image)
    # Perform quantization (reduce the number of bits per channel)
    bits_per_channel = 4  # Change this value to adjust the quantization level
    quantized_image = np.uint8(np.floor(image_float / 256 * (2 ** bits_per_channel)) * (256 / (2 ** bits_per_channel)))

    # Convert quantized image to bytes
    quantized_image_bytes = quantized_image.tobytes()

    # Compress the quantized image data using gzip
    compressed_data = gzip.compress(quantized_image_bytes)

    # Save the compressed quantized image
    compressed_image_path = r"C:\Users\97252\Desktop\Copressrion_examples\quantized_image.gz"
    with open(compressed_image_path, 'wb') as f_out:
        f_out.write(compressed_data)

    # Get the size of the compressed image
    compressed_size = os.path.getsize(compressed_image_path)
    print("=============================================")
    print("For Quantized type compression")
    print("Compressed image size:", compressed_size, "bytes")


def AdditionOfAnchors(image_path):
    image = cv2.imread(image_path)
    # Assume you have anchor points defined as (x, y) coordinates
    anchor_points = [(100, 100), (200, 200), (300, 300)]
    # Convert the image to a format suitable for quantization
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = np.float32(image)
    # Perform quantization (reduce the number of bits per channel)
    bits_per_channel = 4  # Change this value to adjust the quantization level
    # Modify quantization to consider anchor points
    quantized_image = np.zeros_like(image_float)
    for anchor in anchor_points:
        x, y = anchor
        quantized_image[x - 10:x + 10, y - 10:y + 10] = np.uint8(
            np.floor(image_float[x - 10:x + 10, y - 10:y + 10] / 256 * (2 ** bits_per_channel)) * (
                    256 / (2 ** bits_per_channel)))

    # Save the quantized image
    quantized_image_path = r"C:\Users\97252\Desktop\Copressrion_examples\quantized_image.jpg"
    cv2.imwrite(quantized_image_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
    # Get the size of the compressed image
    compressed_size = os.path.getsize(quantized_image_path)
    #
    # image_with_anchors_path = quantized_image_path.split(".")[0] + "_with_anchors.jpg"
    # os.rename(quantized_image_path, image_with_anchors_path)
    print("=============================================")
    print("For Quantized image with anchor type compression")
    print("Compressed image size:", compressed_size, "bytes")


def bfs(image, start, anchor_points=None, directions=None):
    if anchor_points is None:
        anchor_points = []  # Initialize anchor points if not provided
    if directions is None:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Default directions

    # Initialize queue for BFS and set for visited pixels
    queue = deque([start])
    visited = set([start])

    # Perform BFS traversal
    traversal_path = []
    while queue:
        current = queue.popleft()
        traversal_path.append(current)  # Store the traversal path

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))

    return traversal_path

def BFSCompression(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    start_pixel = (0, 0)  # Starting pixel
    anchor_points = [(100, 100), (200, 200), (300, 300)]  # Example anchor points
    traversal_path = bfs(image, start_pixel, anchor_points)

    # Convert traversal path to string representation
    encoded_path = ','.join(f"{point[0]},{point[1]}" for point in traversal_path)

    # Save compressed data to file
    with open(output_path, 'w') as file:
        file.write(encoded_path)

    # Print the size of the encoded path (after compression)
    compressed_size = len(encoded_path.encode('utf-8'))  # Size in bytes
    print("Size after compression:", compressed_size, "bytes")



def greedy_compression(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Define directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Example usage:
    start_pixel = (0, 0)  # Starting pixel
    initial_anchor_points = [(100, 100), (200, 200), (300, 300)]  # Initial anchor points
    traversal_path = bfs(image, start_pixel, initial_anchor_points, directions)
    reconstructed_image = np.zeros_like(image)  # Placeholder for the reconstructed image

    # Calculate error between original and reconstructed images
    error = np.sum(np.abs(image - reconstructed_image))

    # Greedily select additional anchor points
    target_compression_level = 0.8  # Example target compression level
    while error > target_compression_level:
        # Select additional anchor point greedily based on error reduction
        # Update traversal path, reconstruct image, and error calculation

        # For demonstration, let's assume we add an anchor point at (400, 400)
        additional_anchor = (400, 400)
        initial_anchor_points.append(additional_anchor)
        traversal_path = bfs(image, start_pixel, initial_anchor_points, directions)
        reconstructed_image = np.zeros_like(image)  # Placeholder for the reconstructed image
        error = np.sum(np.abs(image - reconstructed_image))


def Opengzip(input_file_path,output_file_path):
    input_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\gzip"
    output_file_path = r"C:\Users\97252\Desktop\Copressrion_examples\gzip.jpg"
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print("ZIP Decompression completed.")

def Openzlib(input_file_path,output_file_path):

    try:
        with open(input_file_path, 'rb') as f_in:
            compressed_data = f_in.read()
        decompressed_data = zlib.decompress(compressed_data)
        with open(output_file_path, 'wb') as f_out:
            f_out.write(decompressed_data)
        print("LIB Decompression completed.")
    except zlib.error as e:
        print("Error occurred during zlib decompression:", e)


def BZIP2_Decompression(compressed_file_path, output_file_path):
    with open(compressed_file_path, 'rb') as f_in:
        compressed_data = f_in.read()

    decompressed_data = bz2.decompress(compressed_data)

    with open(output_file_path, 'wb') as f_out:
        f_out.write(decompressed_data)

    print("BZIP2 Decompression completed.")


def BFSDecompression(input_path, output_image_path, image_shape):
    # Read the compressed data from the file
    with open(input_path, 'r') as file:
        compressed_data = file.read()

    # Convert the compressed string back to a list of points
    traversal_path = []
    for point_str in compressed_data.split(','):
        coordinates = point_str.split()
        if len(coordinates) == 2:
            try:
                x, y = map(int, coordinates)  # Split and convert to integers
                traversal_path.append((x, y))
            except ValueError:
                print(f"Skipping invalid point: {point_str}")
        else:
            print(f"Skipping malformed point: {point_str}")

    # Initialize an empty image
    reconstructed_image = np.zeros(image_shape, dtype=np.uint8)

    # Set the pixels corresponding to the traversal path
    for point in traversal_path:
        x, y = point
        # Ensure that the point is within the image bounds
        if 0 <= x < image_shape[0] and 0 <= y < image_shape[1]:
            reconstructed_image[x, y] = 255  # Set pixel value to 255 (white)

    # Write the reconstructed image to disk
    cv2.imwrite(output_image_path, reconstructed_image)



if __name__ == "__main__":
    # Read the 3D image
    image_path = r"C:\Users\97252\Desktop\Copressrion_examples\splinter.jpg"
    with open(image_path, 'rb') as f_in:
        uncompressed_data = f_in.read()
    print("Image size before compression:", len(uncompressed_data), "byte")
    ipgz_Compression(uncompressed_data)
    zlib_Compression(uncompressed_data)
    BZIP2_Compression(uncompressed_data)
    LP_transformation(image_path, uncompressed_data)
    Quantized(image_path)
    AdditionOfAnchors(image_path)
    BFSCompression(image_path,r"C:\Users\97252\Desktop\Copressrion_examples\BFS")
    #greedy_compression(image_path)
    Opengzip(r"C:\Users\97252\Desktop\Copressrion_examples\gzip",r"C:\Users\97252\Desktop\Copressrion_examples\gzip.jpg")
    Openzlib(r"C:\Users\97252\Desktop\Copressrion_examples\zlib",r"C:\Users\97252\Desktop\Copressrion_examples\zlib")
    BZIP2_Decompression(r"C:\Users\97252\Desktop\Copressrion_examples\BZIP2", r"C:\Users\97252\Desktop\Copressrion_examples\BZIP2.jpg")
    image_shape = (400, 400)  # Provide the shape of the original image
    BFSDecompression(r"C:\Users\97252\Desktop\Copressrion_examples\BFS", r"C:\Users\97252\Desktop\Copressrion_examples\BFS.jpg", image_shape)
