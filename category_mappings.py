# Usage:
# Give filename fname
# Get mapping from index to category and mapping from category to index
def get_mappings(fname = 'categories.txt'):
    fname = 'categories.txt'
    with open(fname) as f:
        content = f.readlines()
    category_to_index = dict()
    index_to_category = dict()
    for i in range(len(content)):
        index_to_category[i] = content[i].strip()
        category_to_index[content[i].strip()] = i
    return index_to_category, category_to_index

