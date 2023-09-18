
from datetime import timedelta
import pandas as pd
from multiprocessing import Pool

# Function to generate single count features for given time windows and columns
def generate_single_count_features_for_chunk(df_chunk, time_windows, columns):
    feature_df = df_chunk.copy()
    
    for time_window in time_windows:
        for col in columns:
            new_col_name = f"{col}_{time_window}s"
            feature_df[new_col_name] = 0  # Initialize new feature column with zeros
            
            for i, row in feature_df.iterrows():
                time_threshold = row['click_time'] - timedelta(seconds=time_window)
                mask = (df_chunk['click_time'] >= time_threshold) & (df_chunk['click_time'] <= row['click_time']) & (df_chunk[col] == row[col])
                feature_df.at[i, new_col_name] = df_chunk[mask].shape[0]
                
    return feature_df

# Function to split DataFrame into chunks
def split_dataframe(df, num_chunks):
    chunks = []
    chunk_size = len(df) // num_chunks
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_chunks - 1 else len(df)
        chunks.append(df.iloc[start_index:end_index])
    return chunks

# Main function
def main():
    df = pd.read_csv('adtrackingFraudData.csv')
    df['click_time'] = pd.to_datetime(df['click_time'])
    
    # Define time windows and columns for which to generate features
    time_windows = [1, 3, 10, 30, 60]
    columns = ['ip', 'os', 'app', 'channel', 'device']
    
    # Split DataFrame into chunks
    num_chunks = 4  # Number of chunks
    df_chunks = split_dataframe(df, num_chunks)
    
    # Use multiprocessing to generate features for each chunk
    with Pool(processes=num_chunks) as pool:
        processed_chunks = pool.starmap(generate_single_count_features_for_chunk, [(chunk, time_windows, columns) for chunk in df_chunks])
    
    # Concatenate processed chunks back into a single DataFrame
    feature_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Save the feature DataFrame to a CSV file
    feature_df.to_csv('feature_engineered_clickfraud.csv', index=False)

# Entry point
if __name__ == '__main__':
    main()
