# A Cross-Domain Recommender System using Deep Coupled Autoencoders

Cross-domain recommendation tackles **data sparsity** and **cold-start** by leveraging knowledge across domains.  
This repository introduces two **coupled autoencoder-based deep learning methods** that jointly learn representations and mappings for users/items, achieving superior performance in **extreme cold-start scenarios**.

---

## 📂 Datasets

This repository uses datasets consistent with those described in:  
**A. Gkillas and D. Kosmopoulos, "A cross-domain recommender system using deep coupled autoencoders," ACM Transactions on Recommender Systems, 2025.**

- **Dataset**: [Main Link](https://drive.google.com/drive/folders/1olhKBNVzcfkS2g4E5zdbmlfuo-7G-Edd?usp=sharing)

- **MovieLens & Netflix Subsets** (as reported in *Table 2* of the manuscript).  
  The file names in this repository match exactly the subset numbering in the paper:  
  - `movielens_data_subsets_no1.csv` + `netflix_data_subsets_no1.csv` → Subset No. 1  
  - `movielens_data_subsets_no2.csv` + `netflix_data_subsets_no2.csv` → Subset No. 2  
  - `movielens_data_subsets_no3.csv` + `netflix_data_subsets_no3.csv` → Subset No. 3  
  - `movielens_data_subsets_no4.csv` + `netflix_data_subsets_no4.csv` → Subset No. 4  

- **Douban Subsets** (as reported in *Table 3* of the manuscript).  
  The file names also follow the same naming convention:  
  - `Douban_book_R2.csv` + `Douban_movie_R.csv` → Douban Book / Douban Movie subset  
  - `Douban_music_R.csv` + `Douban_book_R2.csv` → Douban Music / Douban Book subset  

Thus, the dataset file names in this repository **directly correspond to the subset numbering and naming used in Tables 2 and 3 of the paper**, ensuring reproducibility.

---

## 📑 Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{10.1145/3765614,
  author    = {Gkillas, Alexandros and Kosmopoulos, Dimitrios},
  title     = {A cross-domain recommender system using deep coupled autoencoders},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3765614},
  doi       = {10.1145/3765614},
  note      = {Just Accepted},
  journal   = {ACM Trans. Recomm. Syst.},
  month     = sep,
  keywords  = {Cross-domain recommendation systems, coupled autoencoders, latent factor models, deep learning}
}
