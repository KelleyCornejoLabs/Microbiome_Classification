try:
    import pandas as pd
except:
    print("Required package pandas not available")
    exit()

try:
    import argparse
except:
    print("Required package argparse not available")
    exit()

parser = argparse.ArgumentParser(description="This tool creates reformats the data included in the VALENCIA github repository to be acceptable by VALENCIA")

#required arguments
required = parser.add_argument_group("Required arguments")
required.add_argument("-i", "--input", help="Path to input CSV file",required=True)
required.add_argument("-o","--output", help="Output file with Valencia acceptable data")

#reading arguments into parser
args = parser.parse_args()

try:
    data = pd.read_csv(args.input)
except FileNotFoundError:
    print("Invalid path")


cols_to_remove = ['Subject_number','HC_CST', 'HC_subCST','Val_CST','Val_subCST','I-A_sim','I-B_sim','II_sim','III-A_sim','III-B_sim','IV-A_sim','IV-B_sim','IV-C0_sim','IV-C1_sim','IV-C2_sim','IV-C3_sim','IV-C4_sim','V_sim']
for col in cols_to_remove:
    try:
        data = data.drop(columns=[col])
    except: 
        continue
data = data.rename(columns={'total_reads':'read_count', 'Sample_number_for_SRA':'sampleID'})

if args.output != None:
    try:
        data.to_csv(args.output, index=False)
    except FileNotFoundError:
        print("Invalid path")
else:
    data.to_csv("output.csv", index=False)

