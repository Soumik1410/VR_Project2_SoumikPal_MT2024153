import os
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
from transformers.models.blip.modeling_blip_text import BlipTextEmbeddings
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


class CustomBlipForQuestionAnswering(BlipForQuestionAnswering):
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, labels=None, **kwargs):
        if 'inputs_embeds' in kwargs:
            kwargs.pop('inputs_embeds')

        return super().forward(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, **kwargs)


#orig_forward = BlipTextEmbeddings.forward

def patched_forward(self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
    if input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if position_ids is None:
        position_ids = torch.arange(
            past_key_values_length, input_shape[-1] + past_key_values_length, dtype=torch.long, device=self.position_ids.device
        ).unsqueeze(0).expand(input_shape)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    position_embeddings = self.position_embeddings(position_ids)

    embeddings = inputs_embeds + position_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    BlipTextEmbeddings.forward = patched_forward

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_config = PeftConfig.from_pretrained("Soumik1996/blip-vqa-qlora")
    base_model_name = peft_config.base_model_name_or_path
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained("Soumik1996/blip-vqa-qlora")
    # print(base_model_name)
    base_model = CustomBlipForQuestionAnswering.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "Soumik1996/blip-vqa-qlora")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"Total parameters: {total_params:,}")
    #print(f"Trainable parameters: {trainable_params:,}")

    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_name']}"
        question = str(row["question"])

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=1)
            pred_answer = processor.decode(out[0], skip_special_tokens=True).strip().lower()
        except Exception as e:
            pred_answer = "error"
        generated_answers.append(pred_answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()


