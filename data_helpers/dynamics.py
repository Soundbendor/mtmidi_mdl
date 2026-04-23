inflection_marker = "-" # marks dynamic types with inflection

# one dynamic
dyn_cat_one = ["flat"]

# two dynamics but from beginning to end
dyn_cat_two = ["cresc", "decresc"]

dyn_cat_three = ['hairpin', 'revhairpin', 'subp', 'subf']
# two dynamics but can change dynamic inflection point
# name formatL {type}-{start_of_dyn2}
dyn_subcat_three = [f"{x}{inflection_marker}{y}" for x in dyn_cat_three for y in range(1,4)]

dyn_categories = dyn_cat_one + dyn_cat_two + dyn_cat_three
dyn_subcategories = dyn_cat_one + dyn_cat_two + dyn_subcat_three

num_categories = len(dyn_categories)
num_subcategories = len(dyn_subcategories)

dyn_category_to_idx = {x:i for (i,x) in enumerate(dyn_categories)}
dyn_idx_to_category = {i:x for (x,i) in dyn_category_to_idx.items()}
dyn_subcategory_to_idx = {x:i for (i,x) in enumerate(dyn_subcategories)}
dyn_idx_to_subcategory = {i:x for (x,i) in dyn_subcategory_to_idx.items()}

