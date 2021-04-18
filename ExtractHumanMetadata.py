import sys
import os
import json

def generate_csv(json_path, csv_out_path):
    f_in = open(json_path, "r")
    f_out = open(csv_out_path, "w")

    f_out.write("img_path\n")

    for line in open(json_path, "r"):
        j = json.loads(line)
        im_path = j["img"]
        im_path = os.path.join("/home/kyleaj/Thesis/code/OrigPhotos/data", im_path)
        f_out.write(im_path)
        f_out.write("\n")

    f_out.close()
    f_in.close()

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        generate_csv(sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError()