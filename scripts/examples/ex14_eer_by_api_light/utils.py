
def print_title(title):
    print(f"{'='*10} {'='*len(title)} {'='*10}")
    print(f"{'='*10} {title} {'='*10}")
    print(f"{'='*10} {'='*len(title)} {'='*10}")

def dict_info(embedding):
    key_num = len(embedding.keys())
    value_num = 0
    for key in embedding:
        value_num += len(embedding[key])
    print(f"Key num: {key_num}")
    print(f"Value num: {value_num}")
    