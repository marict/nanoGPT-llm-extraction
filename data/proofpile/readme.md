# Proof‑Pile dataset

After running `prepare_proofpile.py` you will get:

* `train.bin` ≈16.6 GB, `val.bin` ≈2.8 GB
* Train tokens ≈6.9 B, validation tokens ≈1.4 B (GPT‑2 tokenization with EOT appended)
* 270 000 training documents and 46 300 validation documents

The raw dataset is `hoskinson-center/proof-pile` on Hugging Face. It contains 8.3 B tokens of mathematical text and code drawn from arXiv, proof‑assistant libraries, Stack Exchange, and wiki sources.([huggingface.co](https://huggingface.co/datasets/hoskinson-center/proof-pile))

## References

* Proof‑Pile dataset card on Hugging Face.([huggingface.co](https://huggingface.co/datasets/hoskinson-center/proof-pile))
* Proof‑Pile build repository on GitHub.([github.com](https://github.com/zhangir-azerbayev/proof-pile?utm_source=chatgpt.com))
