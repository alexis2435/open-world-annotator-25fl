import os
print("LOADING APP:", __file__)

from fastapi import FastAPI, UploadFile, File, Form
import tempfile
import base64
import os

from backend.inference.ensemble_inference import (
    run_bounding_box_only,
    run_box_and_segmentation
)

app = FastAPI()

@app.get("/ping")
def ping():
    return {"msg": "alive"}

@app.post("/simple_test")
async def simple_test(file: UploadFile = File(...)):
    print(">>> /simple_test called")

    # 保存临时文件
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(await file.read())
    tmp.close()

    print(">>> Saved temp file:", tmp.name)

    # 推理
    out_path = run_bounding_box_only(tmp.name, prompt="test")
    print(">>> Inference output:", out_path)

    # 转 base64
    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    # 删除输入文件
    os.remove(tmp.name)

    return {
        "image": encoded,
        "input_file": tmp.name,
        "output_file": out_path
    }

@app.post("/infer_single")
async def infer_single(
    file: UploadFile,
    prompt: str = Form(...),
    mode: str = Form("bbox")
):
    image_bytes = await file.read()

    result = run_inference_on_image(
        image_bytes=image_bytes,
        prompt=prompt,
        mode=mode
    )

    return result

@app.post("/infer_batch")
async def infer_batch(
    files: List[UploadFile],
    prompt: str = Form(...),
    mode: str = Form("bbox")
):
    images = [await f.read() for f in files]

    results = run_inference_on_image_list(
        image_list=images,
        prompt=prompt,
        mode=mode
    )

    return {"results": results}

@app.post("/infer_zip")
async def infer_zip(
    file: UploadFile,
    prompt: str = Form(...),
    mode: str = Form("bbox")
):
    # 将 Zip 保存为临时文件
    zip_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    zip_temp.write(await file.read())
    zip_temp.close()

    # 解压到临时目录
    extract_dir = unzip_to_temp(zip_temp.name)

    results = run_inference_on_folder(
        folder_path=extract_dir,
        prompt=prompt,
        mode=mode
    )

    return {"results": results}
