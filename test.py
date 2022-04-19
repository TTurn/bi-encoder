from sentence_transformers import SentenceTransformer, util

path = ""
model = SentenceTransformer(path, device="cuda:1")
model.max_seq_length=32

cnt = 0
total_cnt = 0
with open("/data/test.txt","r")as f, open("result.txt","w")as fw:
    for i, line in enumerate(f):
        if i%1000==0:print(i)
        total_cnt+= 1
        line_lst = line.strip().split("\t")
        embeddings1 = model.encode([line_lst[0]], convert_to_tensor=True)
        embeddings2 = model.encode([line_lst[1]], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2).cpu().numpy()
        flag = "F"
        if float(cosine_scores[0][0]) >= 0.75:
            flag = "T"
        if flag == line_lst[2]:
            cnt += 1
        fw.write(line_lst[0]+"\t"+line_lst[1]+"\t"+line_lst[2]+"\t"+str(cosine_scores[0][0])+"\n")
print(cnt, total_cnt, cnt/total_cnt)
