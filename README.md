        # 欠陥候補（非欠陥）と欠陥（中巣）のデータを分離
        df_ng = df_sampled[df_sampled['defect_label'] == 0]
        df_ok = df_sampled[df_sampled['defect_label'] == 1]
        
        # x軸の範囲を設定
        if var in x_axis_settings:
            start, end, step, minor_step = x_axis_settings[var]
            ax.set_xlim(start, end)
        
        # 欠陥候補（非欠陥）のデータをプロット
        sns.stripplot(data=df_ng, x=var, y='defect_label', color='blue', alpha=0.3, 
                      jitter=True, size=5, ax=ax, dodge=True, zorder=1)
        
        # 欠陥（中巣）のデータをプロット
        sns.stripplot(data=df_ok, x=var, y='defect_label', color='red', alpha=1.0, 
                      jitter=True, size=10, ax=ax, dodge=True, zorder=2)
        
        # タイトルと軸ラベルの設定
        plt.title(f'{var}と欠陥ラベルの関係')
        plt.xlabel(var)
        plt.ylabel('欠陥ラベル')
        
        # x軸の目盛りを設定
        if var in x_axis_settings:
            start, end, step, minor_step = x_axis_settings[var]
            major_ticks = np.arange(start, end + step, step)
            minor_ticks = np.arange(start, end + minor_step, minor_step)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
        
        # y軸の目盛りを設定
        plt.yticks([0, 1], ['欠陥候補（非欠陥） (0)', '欠陥（中巣） (1)'])
        
        # 凡例の設定
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='欠陥候補（非欠陥） (0)', 
                                  markerfacecolor='blue', markersize=10, alpha=0.3),
                           Line2D([0], [0], marker='o', color='w', label='欠陥（中巣） (1)', 
                                  markerfacecolor='red', markersize=15)]
        ax.legend(handles=legend_elements, title='欠陥ラベル')
        
        # グラフの調整
        plt.tight_layout()
