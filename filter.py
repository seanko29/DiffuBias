'''
rule based filtering
'''

import argparse
import pandas as pd
import os

def build_args():
    parser = argparse.ArgumentParser(description='Super Resolution')
    parser.add_argument('--data_name', type=str, default='cmnist', help='dataset name')
    parser.add_argument('--save_dir', type=str, default='/home/work/CL4SRec/aaai/filtered_texts', help='where to save')
    parser.add_argument('--pct', type=str, default='5pct', help='percentage of bias-conflicts in the dataset')
    parser.add_argument('--csv_dir', type=str, default='/home/work/CL4SRec/aaai/wonk/save/blip_cmnist/5pct/cmnist_5pct_image_captioning_results.csv', help='csv directory')
    args = parser.parse_args()
    return args

def filter(data_name, csv_file, output_file, output_dir='.'):
    # Define filter lists based on data_name
    if data_name == 'cmnist':
        filter_list = ['0','1','2','3','4','5','6','7','8','9','one','two','three','four','five','six','seven','eight','nine']
    elif data_name == 'cifar10c':
        filter_list = ['airplane','automobile', 'car', 'bird','cat','deer','dog','frog','horse','ship','truck']
    elif data_name == 'bffhq':
        filter_list = ['young', 'old', 'man', 'woman', 'men', 'women', 'boy', 'girl']

    # Read CSV
    df = pd.read_csv(csv_file)
    filtered_df = pd.DataFrame()

    for index, feature in df.iterrows():
        if filter_list in feature[1]:
        
        if filter_list in feature[2]:

        if filter_list in feature[3]:
        print(feature[1])
        print(feature[2])
        break
    exit()

    print(df.columns)
    print(df['generated_text_1'])
    # Filter rows where any of the generated_text columns contains a word from filter_list
    # filtered_df = df[df['generated_text_1'].str.contains('|'.join(filter_list), case=False, na=False) |
    #                  df['generated_text_2'].str.contains('|'.join(filter_list), case=False, na=False) |
    #                  df['generated_text_3'].str.contains('|'.join(filter_list), case=False, na=False)]

    filtered_df['class_label'] = filtered_df['image_path'].str.split('/').str[-3]

    # filtered_df['generated_text'] = filtered_df['generated_text_1'] + ', ' + \
    #                                filtered_df['generated_text_2'] + ', ' + \
    #                                filtered_df['generated_text_3']

    # Only keep the 'generated_text' column and save to CSV
    output_path = os.path.join(output_dir, output_file)

    # filtered_df[['generated_text']].to_csv(output_path, index=False)
    # Save filtered rows to a new CSV file in the specified directory
    # filtered_df.to_csv(output_path, index=False)
    for label, group in filtered_df.groupby('class_label'):
        output_file = f"filtered_class_{label}.csv"
        output_path = os.path.join(output_dir, output_file)
        group.drop('class_label', axis=1).to_csv(output_path, index=False)
        print(f"Filtered data for class {label} saved to {output_path}")
    print(f"Filtered data saved to {output_path}")

# Example Usage:
# filter('cifar10c', 'input.csv', 'filtered_output.csv', '/desired/directory/path')


if __name__ == '__main__':
    args = build_args()
    data_name = args.data_name
    csv_dir = args.csv_dir
    save_dir = args.save_dir
    pct = args.pct
    filter(data_name, csv_dir, f'filtered_output_{data_name}_{pct}.csv', save_dir)
# Example Usage:
# filter('cifar10c', 'input.csv', 'filtered_output.csv')

    