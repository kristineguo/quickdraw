# Usage:
# Give filename fname
# Get mapping from index to category and mapping from category to index
def get_category_mappings(fname = 'categories.txt'):
    with open(fname) as f:
        content = f.readlines()
    category_to_index = dict()
    index_to_category = []
    for i in range(len(content)):
        category_name = content[i].strip()
        index_to_category.append(category_name)
        category_to_index[category_name] = i
    return index_to_category, category_to_index
