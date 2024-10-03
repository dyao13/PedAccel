import ani
from chunk_data import save_chunk
import os

def process_patient(pat):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    data_files = [f for f in os.listdir(data_dir) if f.startswith(f'Patient{pat}_Event_Row')]

    for i in range(len(data_files)):
        file_path = os.path.join(data_dir, data_files[i])
        save_chunk(file_path, os.path.join(output_dir, f'pat{pat}_{i}.csv'))

        ani.main(os.path.join(data_dir, f'pat{pat}_chunk{i}.csv'))

def main():
    process_patient(2)

if __name__ == '__main__':
    main()