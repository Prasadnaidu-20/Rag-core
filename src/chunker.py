
def overLappingChunk(text,chunk_size,overlapping_size):
    start = 0
    chunks = []

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start = start + chunk_size - overlapping_size

    return chunks



