import csv

def tsv2csv(tsv_file,csv_file):
    # 打开TSV文件进行读取，并写入到CSV文件
    with open(tsv_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', encoding='utf-8', newline='') as outfile:
        # 使用csv.reader读取TSV文件，指定分隔符为'\t'
        tsv_reader = csv.reader(infile, delimiter='\t')

        # 使用csv.writer写入CSV文件
        csv_writer = csv.writer(outfile)

        # 将每一行从TSV写入到CSV
        for row in tsv_reader:
            csv_writer.writerow(row)

if __name__ == '__main__':
    tsv2csv('soure_path','output_path')
