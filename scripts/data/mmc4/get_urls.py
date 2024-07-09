from multiprocessing import Pool
import tqdm
import argparse
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument('--input_jsonl', type=str, default=None, help='Local path to the input jsonl file')
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

def gather_image_info(input_jsonl):
    """Gather image info from the input jsonl"""
    # data = []
    output_urls = []
    with open(input_jsonl) as f:
        for line in tqdm.tqdm(f):
            info = json.loads(line.strip())
            for img_item in info['image_info']:
                # data.append({
                #     'local_identifier': img_item['image_name'],
                #     'url': img_item['raw_url'],
                # })
                output_urls.append(img_item['raw_url'])
    # return data
    return output_urls

filename = os.path.basename(args.input_jsonl)
output_filepath = os.path.join(args.output_dir, filename.replace('.jsonl', '_urls.txt'))
print(f'Reading from {args.input_jsonl} and writing to {output_filepath}')
# data = gather_image_info(args)
urls = gather_image_info(args.input_jsonl)
# with open(output_filepath, 'w') as f:
#     for url in urls:
#         f.write(url + '\n')
# save to parquet
import pandas as pd
df = pd.DataFrame({'url': urls})
df.to_parquet(output_filepath.replace('.txt', '.parquet'))
