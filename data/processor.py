import os
import multiprocessing
from functools import partial
import enum
from datasets import Dataset as HFDataset, Features, Value

# --- OPENWEBTEXT PROCESSOR CONSTANTS ---
WINDOW_SIZE = 64
GARBAGE_THRESHOLD = 10
TEXT_THRESHOLD = 45
IS_PRINTABLE_ASCII = bytearray(256)
for i in range(32, 127): IS_PRINTABLE_ASCII[i] = 1

class State(enum.Enum):
    READING_TEXT = 1
    SKIPPING_GARBAGE = 2

def _strip_bytes(byte_slice: bytes) -> bytes:
    start = 0
    end = len(byte_slice)
    for i in range(len(byte_slice)):
        if not byte_slice[i:i+1].isspace():
            start = i
            break
    else: return b''
    for i in range(len(byte_slice) - 1, -1, -1):
        if not byte_slice[i:i+1].isspace():
            end = i + 1
            break
    return byte_slice[start:end]

def _process_chunk_smart(chunk_data: bytes, min_doc_length: int) -> list[str]:
    documents = []
    state = State.SKIPPING_GARBAGE
    doc_start_index = 0
    current_pos = 0
    while current_pos + WINDOW_SIZE <= len(chunk_data):
        window = chunk_data[current_pos : current_pos + WINDOW_SIZE]
        printable_count = sum(IS_PRINTABLE_ASCII[b] for b in window)
        if state == State.SKIPPING_GARBAGE:
            if printable_count > TEXT_THRESHOLD:
                state = State.READING_TEXT
                doc_start_index = max(0, current_pos)
        elif state == State.READING_TEXT:
            if printable_count < GARBAGE_THRESHOLD:
                doc_slice = chunk_data[doc_start_index:current_pos]
                stripped_slice = _strip_bytes(doc_slice)
                if len(stripped_slice) > min_doc_length:
                    try:
                        text = stripped_slice.decode('utf-8')
                        documents.append(text)
                    except UnicodeDecodeError: pass
                state = State.SKIPPING_GARBAGE
        current_pos += WINDOW_SIZE
    if state == State.READING_TEXT:
        doc_slice = chunk_data[doc_start_index:]
        stripped_slice = _strip_bytes(doc_slice)
        if len(stripped_slice) > min_doc_length:
            try:
                text = stripped_slice.decode('utf-8')
                documents.append(text)
            except UnicodeDecodeError: pass
    return documents

class OpenWebTextProcessor:
    def __init__(self, file_path, chunk_Size = 64 * 1024 * 1024, min_doc_length = 200, num_workers = None):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file tại {file_path}.")
        self.file_path = file_path
        self.chunk_Size = chunk_Size
        self.min_doc_length = min_doc_length
        self.numworkes = num_workers if num_workers else os.cpu_count()
        self.dataset = None
        print(f"Sẽ sử dụng {self.numworkes} worker với giải thuật python lọc kĩ tự lạ")
    
    def _generate_chunks_bytes(self):
        with open(self.file_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_Size)
                if not chunk: break
                yield chunk

    def _generate_documents_parallel(self):
        with multiprocessing.Pool(self.numworkes) as pool:
            chunk_genarator = self._generate_chunks_bytes()
            worker_func = partial(_process_chunk_smart, min_doc_length= self.min_doc_length)
            map_results = pool.imap_unordered(worker_func, chunk_genarator)
            for doc_list in map_results:
                for doc_text in doc_list:
                    yield {'text': doc_text}
    
    def build_dataset(self):
        print("Bắt đầu tạo Dataset từ generator song song...")
        features = Features({'text': Value('string')})
        self.dataset = HFDataset.from_generator(self._generate_documents_parallel, features=features)
        print("\nHoàn thành việc tạo Dataset!")
