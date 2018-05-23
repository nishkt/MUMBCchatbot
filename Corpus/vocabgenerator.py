import os

AUG0_FOLDER = "mumbc"

VOCAB_FILE = "vocab.txt"
MUMBC_FILE = "mumbcchatbot.txt"

def generate_vocab_file(corpus_dir):
    """
    Generate the vocab.txt file for the training and prediction/inference. 
    Manually remove the empty bottom line in the generated file.
    """
    vocab_list = []

    # Special tokens, with IDs: 0, 1, 2
    for t in ['_unk_', '_bos_', '_eos_']:
        vocab_list.append(t)

    # The word following this punctuation should be capitalized in the prediction output.
    for t in ['.', '!', '?']:
        vocab_list.append(t)

    # The word following this punctuation should not precede with a space in the prediction output.
    for t in ['(', '[', '{', '``', '$']:
        vocab_list.append(t)

    temp_dict = {}  # A temp dict
    mumbc_file = os.path.join(corpus_dir, AUG0_FOLDER, MUMBC_FILE)
    if os.path.exists(mumbc_file):
        with open(mumbc_file, 'r') as f1:
            for line in f1:
                ln = line.strip()
                if not ln:
                    continue
                if ln.startswith("Q:") or ln.startswith("A:"):
                    tokens = ln[2:].strip().split(' ')
                    for token in tokens:
                        if len(token) and token != ' ':
                            t = token.lower()
                            if t not in vocab_list:
                                if ln.startswith("A:"):  # Keep all for responses
                                    vocab_list.append(t)
                                else:
                                    if t not in temp_dict:
                                        temp_dict[t] = 1
                                    else:
                                        temp_dict[t] += 1
                                        if temp_dict[t] >= 2:
                                            vocab_list.append(t)

    with open(VOCAB_FILE, 'a') as f_voc:
        for v in vocab_list:
            f_voc.write("{}\n".format(v))

    print("Vocab size after mumbc data file scanned: {}".format(len(vocab_list)))

if __name__ == "__main__":
    corp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    generate_vocab_file(corp_dir)
