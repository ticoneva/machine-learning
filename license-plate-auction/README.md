# Predicting License Plate Auction Price with Deep Recurrent Neural Network

### Prerequisites
Python 3, [scipy](https://www.scipy.org/), 
[scikit-learn](http://scikit-learn.org/) and [neon](https://github.com/NervanaSystems/neon).

### Targets
The follow targets are available. The "(D)" suffix means "within-day".
<table>
  <tr>
    <th>y_choice</th>
    <th>Target</th>
  </tr>
  <tr>
    <td>0</td>
    <td>price</td>
  </tr>
  <tr>
    <td>1</td>
    <td>price, categorized in 10 bins</td>
  </tr>
  <tr>
    <td>2</td>
    <td>price, categorized in 100 bins</td>
  </tr>
  <tr>
    <td>3</td>
    <td>unsold indicator</td>
  </tr>
  <tr>
    <td>4</td>
    <td>average price (D)</td>
  </tr>
  <tr>
    <td>5</td>
    <td>standard deviation of price (D)</td>
  </tr>
  <tr>
    <td>6</td>
    <td>median price (D)</td>
  </tr>
  <tr>
    <td>7</td>
    <td>above median price (D)</td>
  </tr>
  <tr>
    <td>8</td>
    <td>above double of median price (D)</td>
  </tr>
  <tr>
    <td>9</td>
    <td>above triple of median price (D)</td>
  </tr>
  <tr>
    <td>10</td>
    <td>price, categorized in 5 bins (D)</td>
  </tr>
  <tr>
    <td>11</td>
    <td>price, categorized in 10 bins (D)</td>
  </tr>
  <tr>
    <td>12</td>
    <td>price, categorized in 20 bins (D)</td>
  </tr>
  <tr>
    <td>13</td>
    <td>standardized price (D)</td>
  </tr>
  <tr>
    <td>14</td>
    <td>price to median ratio (D)</td>
  </tr>
  <tr>
    <td>15</td>
    <td>log price</td>
  </tr>
</table>

### Running the Scripts
Recreate Section 5 and 6.1 of the paper:

```
python plate_whole_day_study.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "wd_y15.csv"
```

Recreate Section 6.2 of the paper:

```
python plate_overtime_noretrain.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "nt_y15.csv"
python plate_overtime_year.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "yt_y15.csv"
python plate_overtime_month.py --y_choice 15 --y_linear 1 --learn_rate 0.001 -z 2048 -e 40 --runs 30 --result_path "mt_y15.csv"
```