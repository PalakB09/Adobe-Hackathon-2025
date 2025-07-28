import os
import subprocess
import threading

# Define the collections and their respective input/output paths
collections = [
    "Collection 1",
    "Collection 2",
    "Collection 3"
]

# Docker image name
docker_image = "pdf-processor"

# Command runner function
def run_docker_for_collection(collection_name):
    base_path = os.path.abspath(collection_name)
    input_json = os.path.join(base_path, "challenge1b_input.json")
    pdfs_dir = os.path.join(base_path, "PDFs")
    output_dir = base_path
    output_json = os.path.join("output", "challenge1b_output.json")

    command = [
        "docker", "run", "--rm",
        "-v", f"{input_json}:/app/challenge1b_input.json",
        "-v", f"{pdfs_dir}:/app/PDFs",
        "-v", f"{output_dir}:/app/output",
        docker_image,
        "python", "document_processor.py",
        "--input", "challenge1b_input.json",
        "--output", output_json
    ]

    print(f"▶ Running Docker for: {collection_name}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {collection_name} completed successfully.")
    else:
        print(f"❌ {collection_name} failed.")
        print(result.stderr)


# Launch each collection in a separate thread
threads = []
for col in collections:
    thread = threading.Thread(target=run_docker_for_collection, args=(col,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for t in threads:
    t.join()

print("All collections processed.")
