import os
from PIL import Image
from collections import deque
from adodbapi.ado_consts import directions
import numpy as np
import zstandard as zstd
import gzip
import shutil
import zlib
import bz2
import lzma
import zstandard as zstd
import cv2


def ipgz_Compression(data):
    compressed_file_path = r"C:\mesh-com\gzip"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    compressed_size = os.path.getsize(compressed_file_path)
    print("=============================================")
    print("For gzip type compression")
    print("Image size after compression:", compressed_size, "byte")


def zlib_Compression(data):
    compressed_file_path = r"C:\mesh-com\zlib"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)

    # Measure the size of the image before compression
    compressed_data = zlib.compress(data)
    # Measure the size of the compressed image
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For zlib type compression")
    print("Compressed size:", compressed_size, "bytes")


def BZIP2_Compression(data):
    compressed_file_path = r"C:\mesh-com\BZIP2"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    # Measure the size of the image before compression
    compressed_data = bz2.compress(data)
    # Measure the size of the compressed image
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For BZIP2 type compression")
    print("Compressed size:", compressed_size, "bytes")


def LZMA_Compression(data):
    compressed_file_path = r"C:\mesh-com\LZMA"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    # Measure the size of the image before compression
    compressed_data = lzma.compress(data)
    # Measure the size of the compressed image
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For LZMA type compression")
    print("Compressed size:", compressed_size, "bytes")


def Zstd_Compression(data):
    compressed_file_path = r"C:\mesh-com\Zstd"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    # Compress the image using Zstandard
    cctx = zstd.ZstdCompressor()
    compressed_data = cctx.compress(data)
    # Measure the size of the compressed image
    compressed_size = len(compressed_data)
    print("=============================================")
    print("For Zstd type compression")
    print("Compressed size:", compressed_size, "bytes")


def LP_transformation(image_path, data):
    compressed_file_path = r"C:\mesh-com\LP"
    with gzip.open(compressed_file_path, 'wb') as f_out:
        f_out.write(data)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    # Apply Laplacian transformation
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Convert the Laplacian result back to uint8
    laplacian_uint8 = np.uint8(np.absolute(laplacian))
    # Save the Laplacian-transformed image
    output_image_path = r"C:\mesh-com\laplacian_transformed_image.jpg"
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
    # Save the quantized image
    quantized_image_path = r"C:\mesh-com\quantized_image.jpg"
    cv2.imwrite(quantized_image_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
    # Get the size of the compressed image
    compressed_size = os.path.getsize(quantized_image_path)
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
    quantized_image_path = r"C:\mesh-com\quantized_image.jpg"
    cv2.imwrite(quantized_image_path, cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR))
    # Get the size of the compressed image
    compressed_size = os.path.getsize(quantized_image_path)
    #
    # image_with_anchors_path = quantized_image_path.split(".")[0] + "_with_anchors.jpg"
    # os.rename(quantized_image_path, image_with_anchors_path)
    print("=============================================")
    print("For Quantized image with anchor type compression")
    print("Compressed image size:", compressed_size, "bytes")


# def bfs(image, start, anchor_points):
#     # Define directions: up, down, left, right
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     direction_symbols = ['U', 'D', 'L', 'R']
#
#     # Initialize queue for BFS and set for visited pixels
#     queue = deque([start])
#     visited = set([start])
#
#     # Perform BFS traversal
#     traversal_path = []
#     while queue:
#         current = queue.popleft()
#         traversal_path.append(current)  # Store the traversal path
#
#         x, y = current
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and (nx, ny) not in visited:
#                 queue.append((nx, ny))
#                 visited.add((nx, ny))
#
#     return traversal_path
#
# def BFSCompression(image_path):
#
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     start_pixel = (0, 0)  # Starting pixel
#     anchor_points = [(100, 100), (200, 200), (300, 300)]  # Example anchor points
#     traversal_path = bfs(image, anchor_points, start_pixel)
#
#     # Convert traversal path to string representation
#     direction_symbols = ['U', 'D', 'L', 'R']
#     encoded_path = ''.join(direction_symbols[directions.index(direction)] for direction in traversal_path)
#
#     # Print the size of the encoded path (after compression)
#     compressed_size = len(encoded_path.encode('utf-8'))  # Size in bytes
#     print("Size after compression:", compressed_size, "bytes")

# def bfs(image, start, anchor_points):
#     # Initialize queue for BFS and set for visited pixels
#     queue = deque([start])
#     visited = set([start])
#
#     # Perform BFS traversal
#     traversal_path = []
#     while queue:
#         current = queue.popleft()
#         traversal_path.append(current)  # Store the traversal path
#
#         x, y = current
#         for dx, dy in directions:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and (nx, ny) not in visited:
#                 queue.append((nx, ny))
#                 visited.add((nx, ny))
#
#     return traversal_path
#
# def reconstruct_image(traversal_path, anchor_points):
#     # Reconstruct the image from the traversal path and anchor points
#     reconstructed_image = np.zeros_like(image)
#     for point in traversal_path:
#         x, y = point
#         reconstructed_image[x, y] = 255  # Set pixel value (for demonstration)
#
#     # Add anchor points to the reconstructed image
#     for anchor in anchor_points:
#         x, y = anchor
#         reconstructed_image[x, y] = 255  # Set pixel value (for demonstration)
#
#     return reconstructed_image
#
# def greedy_compression(image_path):
#
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Define directions: up, down, left, right
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#     direction_symbols = ['U', 'D', 'L', 'R']
#     # Example usage:
#     start_pixel = (0, 0)  # Starting pixel
#     initial_anchor_points = [(100, 100), (200, 200), (300, 300)]  # Initial anchor points
#     traversal_path = bfs(image, start_pixel, initial_anchor_points)
#     reconstructed_image = reconstruct_image(traversal_path, initial_anchor_points)
#
#     # Calculate error between original and reconstructed images
#     error = np.sum(np.abs(image - reconstructed_image))
#
#     # Greedily select additional anchor points
#     target_compression_level = 0.8  # Example target compression level
#     while error > target_compression_level:
#         # Select additional anchor point greedily based on error reduction
#         # Update traversal path, reconstruct image, and error calculation
#
#         # For demonstration, let's assume we add an anchor point at (400, 400)
#         additional_anchor = (400, 400)
#         initial_anchor_points.append(additional_anchor)
#         traversal_path = bfs(image, start_pixel, initial_anchor_points)
#         reconstructed_image = reconstruct_image(traversal_path, initial_anchor_points)
#         error = np.sum(np.abs(image - reconstructed_image))

def Opengzip():
    input_file_path = r"C:\mesh-com\gzip"
    output_file_path = r"C:\mesh-com\gzip.jpg"
    with gzip.open(input_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print("Decompression completed.")

def Openzlib():

    input_file_path = "C:/mesh-com/zlib"
    output_file_path = "C:/mesh-com/zlib.jpg"
    try:
        with open(input_file_path, 'rb') as f_in:
            compressed_data = f_in.read()
        decompressed_data = zlib.decompress(compressed_data)
        with open(output_file_path, 'wb') as f_out:
            f_out.write(decompressed_data)
        print("Decompression completed.")
    except zlib.error as e:
        print("Error occurred during zlib decompression:", e)



if __name__ == "__main__":
    # Read the 3D image
    image_path = r"C:\mesh-com\pic.jpg"
    with open(image_path, 'rb') as f_in:
        uncompressed_data = f_in.read()
    print("Image size before compression:", len(uncompressed_data), "byte")
    ipgz_Compression(uncompressed_data)
    zlib_Compression(uncompressed_data)
    BZIP2_Compression(uncompressed_data)
    LZMA_Compression(uncompressed_data)
    Zstd_Compression(uncompressed_data)
    LP_transformation(image_path, uncompressed_data)
    Quantized(image_path)
    AdditionOfAnchors(image_path)
    # BFSCompression(image_path)
    # greedy_compression(image_path)
    Opengzip()
    Openzlib()
