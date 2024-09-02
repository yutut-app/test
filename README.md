# 品番ごとの固定色を定義
color_map = {
    2: 'red',
    4: 'blue',
    5: 'green',
    6: 'purple'
}

                line, = ax.plot(x_values, y_values, label=f'品番 {product}', marker='o', color=color_map.get(product, 'gray'))
