# Predicting License Plate Auction Price with Deep Recurrent Neural Network

### Prerequisites
Python 3, [scipy](https://www.scipy.org/), 
[scikit-learn](http://scikit-learn.org/) and [Nervana Neon](https://github.com/NervanaSystems/neon).

### Targets
The follow targets are available. The "_d" suffix means "within-day".
| y_choice | Target           |
|----------|------------------|
| 0        | price            |
| 1        | price_cat10      |
| 2        | price_cat100     |
| 3        | unsold           |
| 4        | avg_price_d      |
| 5        | sd_price_d       |
| 6        | median_price_d   |
| 7        | abv_median_d     |
| 8        | dbl_median_d     |
| 9        | tri_median_d     |
| 10       | price_cat5_d     |
| 11       | price_cat10_d    |
| 12       | price_cat20_d    |
| 13       | p_std_d          |
| 14       | p_median_ratio_d |
| 15       | ln_price         |

### Running the Scripts
Recreate Section 4 and 5.1 of the paper:

```python
python plate_whole_day_study.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "wd_y15.csv"
```

Recreate Section 5.2 of the paper:

```python
python plate_overtime_noretrain.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "nt_y15.csv"
python plate_overtime_year.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "yt_y15.csv"
python plate_overtime_month.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "mt_y15.csv"
```