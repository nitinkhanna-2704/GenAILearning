## Tokenizers: BPE, WordPiece, SentencePiece

LLMS are built on tokens not words. Tokens are not exactly words but can be called as pairs or group of characters put together based on a function like likelihood or frequency to keep the vocab size limited and convert language to numbers.


Considering the whole word as a token for all the words in a language will lead to huge cost due to millions of words similarly each character cannot be considered as a separate token. 
Hence techniques like <b>Subword tokenization<b> is done to make meaningful tokens and also keep the VOCAB_SIZE finite for a model.

### Byte Pair Tokenization (BPE)
BPE is a greedy compression algorithm repurposed for tokenization. The idea is simple. Combinining the characters occuring together until VOCAB_SIZE == Threshold

Start with individual characters. Count every adjacent pair in the training corpus. Merge the most commonly occuring pair into a new token. Repeat until you reach your target vocabulary size.

graph LR
    subgraph Training["BPE Training Loop"]
        direction TB
        T1["Start: character vocabulary"] --> T2["Count all adjacent pairs"]
        T2 --> T3["Merge most frequent pair"]
        T3 --> T4["Add merged token to vocab"]
        T4 --> T5{"Reached target\nvocab size?"}
        T5 -->|No| T2
        T5 -->|Yes| T6["Done: save merge table"]
    end


### WordPeice
This is similar to BPE but instead of frequency it uses <b>LIKELIHOOD<b>

Merge Criteria for a pair XY = count of (X and Y)  / (count(x) * count(Y))

### Sentence Piece 

It considers everything to be single string even spaces and special chars. it operates in BPE mode or unigram mode: BPE is bottom up approach and Unigram is top down approach. In unigram chars which effect the likelihood least are trimmed in each cycle until vocab size is reached/


## Problem of Multilinguality

The tokenizers of one language penalize the words of other languages heavily as they try to brak the words into simple words like unhappiness into un , happi and ness. 


![alt text](image.png)


Sennrich et al., 2016 -- "Neural Machine Translation of Rare Words with Subword Units" -- the paper that introduced BPE for NLP, turning a 1994 compression algorithm into the foundation of modern tokenization


A standard tokenizer that can be used in production has 5 steps which are followed fter receiving raw text:

Raw Text-> Normalize -> Pretokenize->BPE Merge-> Special Token->Token Ids

Each stage has a specific task

<li> Normalize : NFKC Unicode, lowercase optional, strip accents optional. "fi" ligature (U+FB01) becomes "fi" (two chars). Without this, same word gets different tokens.

<li> Pre-Tokenize:Split text into chunks before BPE.	Prevents BPE from merging across word boundaries. "the cat" should never produce a token "e c".

<li> BPE Merge : Apply learned merge rules to byte sequences. The core compression. Turns raw bytes into subword tokens.

<li> Special Tokens: Inject EOS, BOS, PAD and other chat complete markers. These tokens have fixed IDs. They never participate in BPE merges. The model needs them for structure.

<li> ID Mapping: Convert token strings to integer IDs	The model sees integers, not strings.



#### Byte-Level BPE
Lesson 01's tokenizer operated on UTF-8 bytes. That was the right call. But we skipped something important: what happens when those bytes are <bold> not valid UTF-8?

Byte-level BPE solves this by treating every possible byte value (0-255) as a valid token. Your base vocabulary is exactly 256 entries. Any file -- text, binary, corrupted -- can be tokenized without producing an unknown token.

GPT-2 added a trick: map each byte to a printable Unicode character so the vocabulary stays human-readable. Byte 0x20 (space) becomes the character "G" in their mapping. This is purely cosmetic. The algorithm does not care.

The real power: byte-level BPE handles every language on earth. Chinese characters are 3 UTF-8 bytes each. Japanese can be 3-4 bytes. Arabic, Devanagari, emoji -- all just byte sequences. The BPE algorithm finds patterns in these byte sequences exactly the same way it finds patterns in English ASCII bytes.

#### Pre-Tokenization
Before BPE touches your text, you need to split it into chunks. This prevents the merge algorithm from creating tokens that span word boundaries.

GPT-2 uses a regex pattern to split text:

```code 
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Llama uses SentencePiece, which skips regex entirely. It treats the raw byte stream as one long sequence and lets the BPE algorithm figure out the boundaries. This is simpler but gives BPE more freedom to create cross-word tokens.

The choice matters. GPT-2's regex prevents the tokenizer from learning that "the" at the end of one word and "the" at the start of the next should merge. SentencePiece allows it, which sometimes produces more efficient compression but less interpretable tokens.

#### Special Tokens
Every production tokenizer reserves token IDs for structural markers:

- [BOS]: Beginning of sequence	Llama 3, GPT
- [EOS]: End of sequence	All models
- [PAD]: Padding for batch alignment	BERT, T5
- [UNK]: Unknown token (byte-level BPE eliminates this)	BERT, WordPiece
- <|im_start|>: Chat message boundary start	ChatGPT, Qwen
- <|im_end|>: Chat message boundary end	ChatGPT, Qwen
- <|user|>:	User turn marker	Llama 3
- <|assistant|>:	Assistant turn marker	Llama 3

Special tokens are never split by BPE. They are matched exactly before the merge algorithm runs, replaced with their fixed ID, and the surrounding text is tokenized normally.


#### Chat Templates

When you send messages to a chat model, the API accepts a list of messages:

[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]

The model does not see JSON. It sees a flat token sequence. The chat template converts messages into that flat sequence using special tokens. Every model does this differently:

Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>

<b> Get the template wrong and the model produces garbage. It was trained on one exact format. Any deviation -- a missing newline, a swapped token, an extra space -- puts the input outside the training distribution. <b>



### Speed

python is too slow to be in production. tiktoken is written in RUST with py bindings. Huggingface is also Rust. SentencePeice is C++

For perspective: tokenizing 15 trillion tokens for Llama 3 pre-training at 1 million tokens per second (fast Python) would take 174 days. At 100 million tokens per second (Rust), it takes 1.7 days.

You are building in Python to understand the algorithm. In production, you would use a compiled implementation and only touch the Python wrapper.
