# %% [markdown]
# # Data extraction notebook
# 
# This notebook extracts the posts data from the Stack .xml archive dumps and saves them to disk with an indicated period.

# %%
# Imports
import contextlib
import gc
import os
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl
import psutil
import py7zr
from lxml import etree

# %% [markdown]
# ## Import data from large archive files:
# https://archive.org/download/stackexchange

# %%
def log_memory_usage(label=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage {label}: {mem_info.rss / 1024 / 1024:.2f} MB")


@contextlib.contextmanager
def temp_chunk_files(temp_dir, files_to_clean=None):
    """Context manager to handle temporary chunk files.

    Creates a temporary directory if it doesn't exist and cleans up all files
    in that directory when exiting the context.

    Parameters:
        temp_dir: Directory where temporary files are stored
        files_to_clean: List of files to clean up (optional)
    """
    # Create the temp directory if it doesn't exist
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Use provided list or create a new one
    files_to_track = files_to_clean or []

    try:
        # Yield the list that will store paths to the temporary files
        yield files_to_track
    finally:
        # Clean up all the temporary files when done
        for file in files_to_track:
            try:
                os.remove(file)
                print(f"Removed temporary file: {file}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {file}: {e}")


def process_xml_in_7z(
    archive_path,
    batch_size=5000,
    start_date=None,
    end_date=None,
    record_tag="row",
    chunk_to_disk=False,
    temp_dir="data/temp/",
    micro_batch_size=100,
):
    """
    Process an XML file within a 7z archive efficiently, optimized for Stack Exchange data.
    Only filters by non-empty titles and date range.

    Args:
        archive_path (str): Path to the .7z archive
        batch_size (int): Number of elements to process in each batch
        start_date (str): Optional start date in format 'YYYY-MM-DD'
        end_date (str): Optional end date in format 'YYYY-MM-DD'
        record_tag (str): XML tag name for records to process (default: "row")
        chunk_to_disk (bool): Whether to write intermediate chunks to disk (for very large files)
        temp_dir (str): Directory to store temporary chunk files if chunking is enabled
        micro_batch_size (int): Size of micro-batches for more frequent memory clearing

    Returns:
        pl.DataFrame: Polars DataFrame containing the processed data with all columns
    """
    print(f"Starting processing of {archive_path}")
    log_memory_usage("at start")

    # Convert date strings to datetime objects if provided
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    # Get the filename inside the archive
    xml_filename = None
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        file_list = archive.getnames()
        if not file_list:
            raise ValueError("No files found in archive")

        # Look for Posts.xml
        for filename in file_list:
            if filename.endswith("Posts.xml"):
                xml_filename = filename
                break

        if not xml_filename:
            # Just use the first file if Posts.xml isn't found
            xml_filename = file_list[0]

    print(f"Processing XML file: {xml_filename}")

    # Use 7z command-line tool to pipe the content without extraction
    cmd = ["7z", "e", "-so", archive_path, xml_filename]
    print(f"Executing: {' '.join(cmd)}")

    # Initialize tracking variables
    all_data = []
    micro_batch = []
    total_processed = 0
    total_skipped = 0
    chunk_files = []
    chunk_count = 0

    # Create temp directory if chunking is enabled
    if chunk_to_disk:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    # Start the extraction process with controlled buffer size
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,  # 1MB buffer
    )

    try:
        # Create iterator with lxml
        context = etree.iterparse(
            process.stdout,
            events=("end",),
            tag=record_tag,
            recover=True,
            huge_tree=True,
            remove_blank_text=True,
            remove_comments=True,
            remove_pis=True,
        )

        for _, elem in context:
            # Only skip if it's not a question (PostTypeId=1)
            post_type = elem.get("PostTypeId")
            if post_type != "1":
                total_skipped += 1
                elem.clear()
                continue

            # Date filter (if specified)
            skip_record = False
            if start_dt or end_dt:
                date_attr = elem.get("CreationDate")
                if date_attr:
                    try:
                        # Parse date for filtering
                        if "T" in date_attr:
                            record_date = datetime.fromisoformat(
                                date_attr.replace("Z", "+00:00")
                            )
                        else:
                            record_date = datetime.strptime(date_attr, "%Y-%m-%d")

                        # Skip if out of date range
                        if (start_dt and record_date.date() < start_dt.date()) or (
                            end_dt and record_date.date() > end_dt.date()
                        ):
                            skip_record = True
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid date format '{date_attr}', error: {e}")

            if skip_record:
                total_skipped += 1
                elem.clear()
                continue

            # If we reach here, the record should be included
            # Extract ALL attributes - no filtering of columns
            row_data = {
                k: str(v) if v is not None else None for k, v in elem.attrib.items()
            }

            # Add to micro-batch
            micro_batch.append(row_data)
            total_processed += 1

            # Clear element to free memory
            elem.clear()
            # Also eliminate previous siblings to keep memory usage low
            while elem.getprevious() is not None:
                del elem.getparent()[0]

            # Process in micro-batches to avoid memory spikes
            if len(micro_batch) >= micro_batch_size:
                all_data.extend(micro_batch)
                micro_batch = []  # Free the micro-batch memory

                # If we've reached full batch size, process the batch
                if len(all_data) >= batch_size:
                    if chunk_to_disk:
                        # Create and save dataframe chunk
                        chunk_df = pl.from_dicts(
                            all_data, infer_schema_length=min(100000, len(all_data))
                        )
                        chunk_file = f"{temp_dir}/chunk_{chunk_count}.parquet"

                        # Write to parquet with compression
                        chunk_df.write_parquet(chunk_file, compression="zstd")
                        chunk_files.append(chunk_file)
                        chunk_count += 1

                        # Clear memory
                        del chunk_df
                        all_data = []  # Free memory
                        gc.collect()

                    if total_processed % 50000 == 0:
                        print(
                            f"Processed {total_processed:,} records, skipped {total_skipped:,}"
                        )
                        log_memory_usage("during processing")

        # Process any remaining data in the micro-batch
        if micro_batch:
            all_data.extend(micro_batch)
            micro_batch = []  # Free memory

        # Process final batch if there's any data left
        if all_data and chunk_to_disk:
            chunk_df = pl.from_dicts(
                all_data, infer_schema_length=min(100000, len(all_data))
            )
            chunk_file = f"{temp_dir}/chunk_{chunk_count}.parquet"
            chunk_df.write_parquet(chunk_file, compression="zstd")
            chunk_files.append(chunk_file)
            del chunk_df
            all_data = []  # Free memory
            gc.collect()

        print("All records processed. Creating final DataFrame...")

    finally:
        # Terminate the subprocess if it's still running
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)  # Wait for process to terminate

    # Final results processing
    print(
        f"Completed. Total processed: {total_processed:,}, skipped: {total_skipped:,}"
    )

    if chunk_to_disk and chunk_files:
        # Save the list of files we want to process
        files_to_process = chunk_files.copy()

        with temp_chunk_files(temp_dir, files_to_process) as _:
            print(f"Reading and combining {len(files_to_process)} chunks from disk")

            # For very large datasets, combine chunks incrementally to prevent memory overload
            if len(files_to_process) > 0:
                # Start with the first chunk
                result_df = pl.read_parquet(files_to_process[0])

                # Process chunks in groups to limit memory usage
                for i, file in enumerate(files_to_process[1:], 1):
                    if i % 10 == 0:
                        print(f"Combining chunk {i}/{len(files_to_process)}")
                        log_memory_usage(f"after {i} chunks")

                    # Read the chunk
                    try:
                        next_df = pl.read_parquet(file)

                        # Combine with result
                        result_df = pl.concat(
                            [result_df, next_df], how="diagonal_relaxed"
                        )

                        # Release memory
                        del next_df
                        if i % 5 == 0:  # Periodically collect garbage
                            gc.collect()
                    except Exception as e:
                        print(f"Error reading chunk {file}: {e}")
                        continue

                print(f"Final dataframe size: {len(result_df):,} rows")
                log_memory_usage("after combining all chunks")
                return result_df
            else:
                print("No chunks were created. Returning empty DataFrame.")
                return pl.DataFrame()

    else:
        # Process in-memory data
        if all_data:
            print(f"Creating DataFrame from {len(all_data):,} records...")
            df = pl.from_dicts(all_data, infer_schema_length=min(100000, len(all_data)))
            # Clear the all_data list to free memory
            all_data = []
            gc.collect()
            return df
        else:
            return pl.DataFrame()


def process_stack_data(
    archive_path,
    output_file="stack_data.parquet",
    start_date=None,
    end_date=None,
    batch_size=5000,
    large_file=False,
    split_output=False,
    max_rows_per_file=1000000,
    temp_dir=None,
):
    """
    High-level function to process Stack Exchange XML data from any community.
    Keeps all columns and only filters out empty titles (for questions) and by date.

    Args:
        archive_path (str): Path to the 7z archive containing XML data
        output_file (str): Path to save the output parquet file
        start_date (str): Optional start date filter in 'YYYY-MM-DD' format
        end_date (str): Optional end date filter in 'YYYY-MM-DD' format
        batch_size (int): Batch size for processing
        large_file (bool): If True, use disk-based chunking for very large files
        split_output (bool): If True, split output into multiple files for memory efficiency
        max_rows_per_file (int): Maximum rows per file when splitting output
        temp_dir (str): Directory to store temporary chunk files

    Returns:
        pl.DataFrame or None: Processed data, or None if split_output is True
    """
    print(f"Processing {archive_path}")
    log_memory_usage("before processing")

    # Set a community-specific temp directory if not provided
    if temp_dir is None:
        community_name = os.path.basename(archive_path).split(".")[0]
        temp_dir = f"data/temp/{community_name}/"

    # Always use chunk_to_disk for large_file processing
    df = process_xml_in_7z(
        archive_path=archive_path,
        batch_size=batch_size,
        start_date=start_date,
        end_date=end_date,
        chunk_to_disk=large_file,
        temp_dir=temp_dir,
    )

    if not df.is_empty():
        print(f"Found {len(df):,} records")
        print(f"Columns: {df.columns}")

        # For very large result sets, split the output into multiple files
        if split_output and len(df) > max_rows_per_file:
            output_base, output_ext = output_file.rsplit(".", 1)
            num_files = (len(df) + max_rows_per_file - 1) // max_rows_per_file

            print(
                f"Splitting output into {num_files} files with max {max_rows_per_file:,} rows each"
            )

            for i in range(num_files):
                start_idx = i * max_rows_per_file
                end_idx = min((i + 1) * max_rows_per_file, len(df))

                # Get slice of dataframe
                part_df = df.slice(start_idx, end_idx - start_idx)

                # Save to file
                part_file = f"{output_base}_part{i + 1}.{output_ext}"
                part_df.write_parquet(part_file, compression="zstd")
                print(f"Saved part {i + 1}/{num_files} to {part_file}")

                # Release memory
                del part_df
                gc.collect()

            # Free the main dataframe memory
            del df
            gc.collect()
            log_memory_usage("after saving split files")
            return None  # Return None since we've split the output
        else:
            # Save to single parquet file
            print(f"Saving data to {output_file}")
            df.write_parquet(output_file, compression="zstd")
            print(f"Data saved to {output_file}")
            log_memory_usage("after saving")
            return df
    else:
        print("No matching records found")
        return df

# %% [markdown]
# ## StackOveflow data
# 
# ### All StackOveflow data

# %%
# Extract StackOverflow data from xml in 7z archive
df = process_stack_data(
    "../data/stackoverflow/stackoverflow.com-Posts.7z",
    output_file="../data/stackoverflow/stackoverflow.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ### StackOverflow largest scripting languages
# 
# The intial StackOverflow dataset from January 2021 to March 2024 contains approx. 4 mln entries - too much to preprocess effectively under time constraints. Therefore, we focus on the largest scripting languages (as these assumedly saw the largest impact of early ChatGPT versions). Nevertheless, we also keep the above extract to run a general DiD.

# %%
df.filter(
    pl.col("Tags").str.contains_any(["|python|", "|r|", "|javascript|", "|php|"])
).write_parquet(
    "../data/stackoverflow/stackoverflow_script.parquet", compression="zstd"
)

# %% [markdown]
# ## Law StackExchange

# %%
df_law = process_stack_data(
    "../data/law/law.stackexchange.com.7z",
    output_file="../data/law/law.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ## Academia StackExchange

# %%
df_ac = process_stack_data(
    "../data/academia/academia.stackexchange.com.7z",
    output_file="../data/academia/academia.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ## Physics StackExchange

# %%
df_ph = process_stack_data(
    "../data/physics/physics.stackexchange.com.7z",
    output_file="../data/physics/physics.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ## SuperUser StackExchange

# %%
df_su = process_stack_data(
    "../data/superuser/superuser.com.7z",
    output_file="../data/superuser/superuser.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ## AskUbuntu StackExchange

# %%
df_au = process_stack_data(
    "../data/askubuntu/askubuntu.com.7z",
    output_file="../data/askubuntu/askubuntu.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)

# %% [markdown]
# ## Math StackExchange

# %%
df_math = process_stack_data(
    archive_path="../data/math/math.stackexchange.com.7z",
    output_file="../data/math/math.parquet",
    batch_size=100000,
    start_date="2021-01-01",
    end_date="2024-12-31",
    large_file=True,
)


