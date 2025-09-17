import os
import h5py
import shutil

def process_subfolders(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # skip the root directory itself

        h5_file_path = None
        for fname in files:
            # print(fname)
            if fname.startswith('oo_3') and fname.endswith('.hdf5'):
                h5_file_path = os.path.join(subdir, fname)
                break

        if h5_file_path is None:
            continue  # no relevant HDF5 file found

        try:
            with h5py.File(h5_file_path, 'r') as h5file:
#                if 'dmft-last' not in h5file:
                if 'dmft-010' not in h5file:
                    raise KeyError
        except (KeyError, OSError):  # dataset missing or corrupt file
#           print(f"Deleting contents of: {subdir}")
           print(f"Deleting : {h5_file_path}")
#           os.remove(h5_file_path)
#           for f in os.listdir(subdir):
#               f_path = os.path.join(subdir, f)
#               try:
#                   if os.path.isfile(f_path) or os.path.islink(f_path):
#                       os.remove(f_path)
#                   elif os.path.isdir(f_path):
#                       shutil.rmtree(f_path)
#               except Exception as e:
#                   print(f"Failed to delete {f_path}: {e}")

if __name__ == "__main__":
    base_directory = "w0_4/"  # change this to your root path
    process_subfolders(base_directory)
