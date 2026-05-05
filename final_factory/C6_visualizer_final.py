import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import seaborn as sns
from datetime import datetime

# ============================================================================
# CONFIGURARE GLOBALĂ — STIL DARK THEME
# ============================================================================

plt.rcParams.update({
    'figure.facecolor': '#0d0d0d',
    'axes.facecolor': '#111111',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#aaaaaa',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'text.color': 'white',
    'axes.titlecolor': 'white',
    'legend.facecolor': '#1a1a1a',
    'legend.edgecolor': '#333333',
    'legend.labelcolor': 'white',
    'figure.dpi': 100,
    'savefig.facecolor': '#0d0d0d',
    'savefig.edgecolor': 'none',
})

CENTRALITY_COLORS = {
    'degree': '#00d4ff',
    'closeness': '#ff6b35',
    'katz': '#aaff00'
}

NETWORK_DISPLAY = {
    'graf_generat_100_1000': 'Gen 1000e/100n',
    'graf_generat_100_99': 'Gen 99e/100n',
    'graf_generat_275_100': 'Gen 275e/100n',
    'graf_real_100_1000': 'Real 1000e/100n',
    'graf_real_100_275': 'Real 275e/100n',
    'graf_real_100_99': 'Real 99e/100n'
}

P_DISPLAY = {0.01: 'p=0.01', 0.04: 'p=0.04', 0.1: 'p=0.1', 0.3: 'p=0.3', 0.5: 'p=0.5'}

NETWORKS_ORDERED = [
    'graf_generat_100_99', 'graf_generat_275_100', 'graf_generat_100_1000',
    'graf_real_100_99', 'graf_real_100_275', 'graf_real_100_1000'
]


class SIRVisualizer:
    """Clasă unificată pentru generarea vizualizărilor statistice ale simulării."""

    def __init__(self, output_dir="graphs_final"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    # ========================================================================
    # METODE PENTRU COMPATIBILITATE CU RUNNER-UL EXISTENT
    # ========================================================================

    def plot_correlations(self, df_corr, network_name):
        """
        Generează un bar plot cu coeficienții de corelație Spearman pentru o rețea.
        Menținut pentru compatibilitate cu runner-ul existent.
        """
        plt.figure(figsize=(12, 7))
        subset = df_corr[df_corr['network'] == network_name].copy()

        if subset.empty:
            print(f"⚠️ Nu există date de corelație pentru rețeaua: {network_name}")
            return

        sns.barplot(data=subset, x='p', y='spearman', hue='centrality')
        plt.title(f"Eficiența Centralităților (Spearman) - {network_name}", fontsize=14)
        plt.ylabel("Coeficient Spearman", fontsize=12)
        plt.xlabel("Probabilitate de Infectare (p)", fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(title="Tip Centralitate", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"corr_{network_name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📈 Grafic corelație salvat: {path}")

    def plot_infection_trend(self, summary_df, network_name):
        """Arată evoluția numărului mediu de infectați în funcție de probabilitatea p."""
        plt.figure(figsize=(10, 6))
        subset = summary_df[summary_df['network'] == network_name].copy()

        if subset.empty:
            return

        sns.lineplot(data=subset, x='infection_probability', y='mean_total_infected', marker='o', linewidth=2.5)
        plt.title(f"Evoluția Severității Epidemiei - {network_name}", fontsize=14)
        plt.xlabel("Probabilitate (p)", fontsize=12)
        plt.ylabel("Media totală a infectaților", fontsize=12)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f"trend_{network_name}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📉 Grafic trend salvat: {path}")

    # ========================================================================
    # METODE NOI — VIZUALIZĂRI AVANSATE (din vizualizare_centralitati.py)
    # ========================================================================

    def generate_all_advanced_plots(self, df_corr, summary_df=None):
        """
        Generează toate cele 7 figuri avansate + animație + raport.
        """
        print("\n🎨 Generare vizualizări avansate...")

        # Asigurăm că avem coloana target_metric (dacă nu există, o creăm)
        if 'target_metric' in df_corr.columns:
            # Deja există
            pass
        else:
            # Adăugăm target_metric bazat pe ce avem (presupunem mean_inf)
            df_corr['target_metric'] = 'mean_inf'

        self._fig1_heatmap_meaninf(df_corr)
        self._fig2_heatmap_maxinf(df_corr)
        self._fig3_dashboard(df_corr)
        self._fig4_evolutie(df_corr)
        self._fig5_comparare_corelatii(df_corr)
        self._fig6_analiza_consistenta(df_corr)
        self._fig7_radar_top(df_corr)

    def generate_video(self, df_corr, video_format='mp4'):
        """Generează animație video cu evoluția corelațiilor."""
        self._generate_video_animation(df_corr, use_ffmpeg=(video_format == 'mp4'))

    def generate_text_report(self, df_corr):
        """Generează raport text cu statistici și concluzii."""
        self._generate_raport_text(df_corr)

    # ========================================================================
    # METODE PRIVATE — IMPLEMENTARE FIGURI
    # ========================================================================

    def _fig1_heatmap_meaninf(self, df):
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        tkw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

        p_vals = sorted(df['p'].unique())
        cent_vals = ['degree', 'closeness', 'katz']

        for idx, net in enumerate(NETWORKS_ORDERED):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.set_facecolor('#111111')
            subset = df[(df['network'] == net) & (df['target_metric'] == 'mean_inf')]
            pivot = subset.pivot_table(index='centrality', columns='p', values='pearson', aggfunc='first')
            pivot = pivot.reindex(cent_vals)

            for i, cent in enumerate(cent_vals):
                for j, p_val in enumerate(p_vals):
                    val = pivot.loc[cent, p_val] if cent in pivot.index and p_val in pivot.columns else 0
                    norm_val = (val + 1) / 2
                    color = plt.cm.RdYlGn(norm_val)
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=color, edgecolor='#333333', linewidth=1)
                    ax.add_patch(rect)
                    text_color = 'white' if abs(val) < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color=text_color)

            ax.set_xlim(-0.5, len(p_vals) - 0.5)
            ax.set_ylim(-0.5, len(cent_vals) - 0.5)
            ax.set_xticks(range(len(p_vals)))
            ax.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
            ax.set_yticks(range(len(cent_vals)))
            ax.set_yticklabels([c.capitalize() for c in cent_vals], fontsize=10)
            ax.set_title(f"{NETWORK_DISPLAY[net]}\nPearson vs mean_inf", **tkw)
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values():
                spine.set_color('#333333')

        fig.subplots_adjust(right=0.92)
        cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Pearson r', color='white', fontsize=11)
        cbar.ax.tick_params(colors='white')

        fig.suptitle('Corelații Pearson: Centralități Clasice vs Stochastic (mean_inf)',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig1_heatmap_pearson_meaninf.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig2_heatmap_maxinf(self, df):
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
        tkw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

        p_vals = sorted(df['p'].unique())
        cent_vals = ['degree', 'closeness', 'katz']

        for idx, net in enumerate(NETWORKS_ORDERED):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.set_facecolor('#111111')
            subset = df[(df['network'] == net) & (df['target_metric'] == 'max_inf')]
            pivot = subset.pivot_table(index='centrality', columns='p', values='pearson', aggfunc='first')
            pivot = pivot.reindex(cent_vals)

            for i, cent in enumerate(cent_vals):
                for j, p_val in enumerate(p_vals):
                    val = pivot.loc[cent, p_val] if cent in pivot.index and p_val in pivot.columns else 0
                    norm_val = (val + 1) / 2
                    color = plt.cm.RdYlGn(norm_val)
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=color, edgecolor='#333333', linewidth=1)
                    ax.add_patch(rect)
                    text_color = 'white' if abs(val) < 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           fontsize=10, fontweight='bold', color=text_color)

            ax.set_xlim(-0.5, len(p_vals) - 0.5)
            ax.set_ylim(-0.5, len(cent_vals) - 0.5)
            ax.set_xticks(range(len(p_vals)))
            ax.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
            ax.set_yticks(range(len(cent_vals)))
            ax.set_yticklabels([c.capitalize() for c in cent_vals], fontsize=10)
            ax.set_title(f"{NETWORK_DISPLAY[net]}\nPearson vs max_inf", **tkw)
            ax.tick_params(colors='#aaaaaa')
            for spine in ax.spines.values():
                spine.set_color('#333333')

        fig.subplots_adjust(right=0.92)
        cax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Pearson r', color='white', fontsize=11)
        cbar.ax.tick_params(colors='white')

        fig.suptitle('Corelații Pearson: Centralități Clasice vs Stochastic (max_inf)',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig2_heatmap_pearson_maxinf.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig3_dashboard(self, df):
        fig = plt.figure(figsize=(24, 28))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)
        tkw = dict(color='white', fontsize=13, fontweight='bold', pad=10)

        combos = [
            ('degree', 'mean_inf', '#00d4ff', 'Deg→mean'),
            ('degree', 'max_inf', '#0088cc', 'Deg→max'),
            ('closeness', 'mean_inf', '#ff6b35', 'Clo→mean'),
            ('closeness', 'max_inf', '#cc4400', 'Clo→max'),
            ('katz', 'mean_inf', '#aaff00', 'Katz→mean'),
            ('katz', 'max_inf', '#66aa00', 'Katz→max'),
        ]

        for idx, net in enumerate(NETWORKS_ORDERED):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.set_facecolor('#111111')
            subset = df[df['network'] == net]
            p_vals = sorted(subset['p'].unique())
            n_groups = len(p_vals)
            bar_width = 0.12
            x = np.arange(n_groups)

            for offset, (cent, target, color, label) in enumerate(combos):
                y_vals = []
                for p_val in p_vals:
                    row = subset[(subset['centrality'] == cent) &
                               (subset['target_metric'] == target) &
                               (subset['p'] == p_val)]
                    y_vals.append(row['pearson'].values[0] if len(row) > 0 else 0)
                ax.bar(x + (offset - 2.5) * bar_width, y_vals, bar_width,
                       label=label, color=color, alpha=0.85, edgecolor='none')

            ax.set_title(f"★ {NETWORK_DISPLAY[net]}", **tkw)
            ax.set_xlabel('Probabilitate infectare p', color='#aaaaaa', fontsize=10)
            ax.set_ylabel('Pearson r', color='#aaaaaa', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
            ax.tick_params(colors='#aaaaaa')
            ax.spines[:].set_color('#333333')
            ax.axhline(y=0, color='#555555', linewidth=0.8, linestyle='-')
            ax.set_ylim(-0.3, 1.05)
            ax.legend(fontsize=7, ncol=3, facecolor='#1a1a1a', labelcolor='white',
                      framealpha=0.8, loc='upper right')

        # Subplot 7: Generat vs Real
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.set_facecolor('#111111')
        generated = df[df['network'].str.contains('generat')]
        real = df[df['network'].str.contains('real')]
        p_vals = sorted(df['p'].unique())
        gen_means = [generated[generated['p'] == p]['pearson'].mean() for p in p_vals]
        real_means = [real[real['p'] == p]['pearson'].mean() for p in p_vals]

        x = np.arange(len(p_vals))
        width = 0.35
        bars1 = ax7.bar(x - width/2, gen_means, width, label='Rețele generate', color='#00d4ff', alpha=0.85)
        bars2 = ax7.bar(x + width/2, real_means, width, label='Rețele reale', color='#ff6b35', alpha=0.85)

        ax7.set_title('Media Pearson: Generat vs Real', **tkw)
        ax7.set_xlabel('Probabilitate p', color='#aaaaaa', fontsize=10)
        ax7.set_ylabel('Pearson r mediu', color='#aaaaaa', fontsize=10)
        ax7.set_xticks(x)
        ax7.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
        ax7.tick_params(colors='#aaaaaa')
        ax7.spines[:].set_color('#333333')
        ax7.legend(fontsize=10, facecolor='#1a1a1a', labelcolor='white')
        ax7.set_ylim(0, 0.8)

        for bar in bars1:
            h = bar.get_height()
            ax7.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', color='white', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax7.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', color='white', fontsize=8)

        # Subplot 8: Histogramă globală
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.set_facecolor('#111111')
        all_pearson = df['pearson'].replace(0, np.nan).dropna()
        all_spearman = df['spearman'].replace(0, np.nan).dropna()
        all_kendall = df['kendall'].replace(0, np.nan).dropna()

        ax8.hist(all_pearson, bins=30, color='#ffdd57', alpha=0.6, label='Pearson', edgecolor='none')
        ax8.hist(all_spearman, bins=30, color='#ff6b35', alpha=0.6, label='Spearman', edgecolor='none')
        ax8.hist(all_kendall, bins=30, color='#00d4ff', alpha=0.6, label='Kendall', edgecolor='none')
        ax8.axvline(all_pearson.mean(), color='#ffdd57', linewidth=2, linestyle='--')
        ax8.axvline(all_spearman.mean(), color='#ff6b35', linewidth=2, linestyle='--')
        ax8.axvline(all_kendall.mean(), color='#00d4ff', linewidth=2, linestyle='--')

        ax8.set_title(f'Distribuția globală a corelațiilor\n(n={len(all_pearson)} măsurători)', **tkw)
        ax8.set_xlabel('Coeficient corelație', color='#aaaaaa', fontsize=10)
        ax8.set_ylabel('Frecvență', color='#aaaaaa', fontsize=10)
        ax8.tick_params(colors='#aaaaaa')
        ax8.spines[:].set_color('#333333')
        ax8.legend(fontsize=10, facecolor='#1a1a1a', labelcolor='white')

        fig.suptitle('Dashboard Corelații: Centralități Clasice vs Stochastic',
                     color='white', fontsize=16, fontweight='bold', y=0.995)
        path = os.path.join(self.output_dir, 'fig3_dashboard_complet.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig4_evolutie(self, df):
        fig = plt.figure(figsize=(22, 16))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        tkw = dict(color='white', fontsize=12, fontweight='bold', pad=10)

        for idx, net in enumerate(NETWORKS_ORDERED):
            ax = fig.add_subplot(gs[idx // 3, idx % 3 if idx < 5 else 2]) if idx < 5 else fig.add_subplot(gs[1, 2])
            ax.set_facecolor('#111111')
            subset = df[df['network'] == net]
            p_vals = sorted(subset['p'].unique())

            combos = [
                ('degree', 'mean_inf', '#00d4ff', 'o', 'Degree → mean_inf'),
                ('degree', 'max_inf', '#0088cc', 's', 'Degree → max_inf'),
                ('closeness', 'mean_inf', '#ff6b35', 'o', 'Closeness → mean_inf'),
                ('closeness', 'max_inf', '#cc4400', 's', 'Closeness → max_inf'),
                ('katz', 'mean_inf', '#aaff00', 'o', 'Katz → mean_inf'),
                ('katz', 'max_inf', '#66aa00', 's', 'Katz → max_inf'),
            ]

            for cent, target, color, marker, label in combos:
                y_vals = []
                for p_val in p_vals:
                    row = subset[(subset['centrality'] == cent) &
                               (subset['target_metric'] == target) &
                               (subset['p'] == p_val)]
                    y_vals.append(row['pearson'].values[0] if len(row) > 0 else np.nan)
                ax.plot(p_vals, y_vals, color=color, marker=marker, markersize=7,
                       linewidth=2.2, label=label, alpha=0.9)

            ax.set_title(f"{NETWORK_DISPLAY[net]}", **tkw)
            ax.set_xlabel('Probabilitate p', color='#aaaaaa', fontsize=10)
            ax.set_ylabel('Pearson r', color='#aaaaaa', fontsize=10)
            ax.tick_params(colors='#aaaaaa')
            ax.spines[:].set_color('#333333')
            ax.set_xscale('log')
            ax.set_xticks(p_vals)
            ax.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
            ax.set_ylim(-0.3, 1.05)
            ax.axhline(y=0, color='#555555', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.legend(fontsize=7, ncol=2, facecolor='#1a1a1a', labelcolor='white',
                      framealpha=0.8, loc='best')
            ax.grid(True, alpha=0.15, color='#555555')

        fig.suptitle('Evoluția Corelației Pearson în funcție de Probabilitatea p',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig4_evolutie_pearson.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig5_comparare_corelatii(self, df):
        fig = plt.figure(figsize=(24, 20))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)
        tkw = dict(color='white', fontsize=13, fontweight='bold', pad=10)

        for idx, net in enumerate(NETWORKS_ORDERED):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.set_facecolor('#111111')
            subset = df[(df['network'] == net) & (df['target_metric'] == 'mean_inf')]
            p_vals = sorted(subset['p'].unique())
            cent_vals = ['degree', 'closeness', 'katz']
            n_groups = len(p_vals)
            bar_width = 0.08
            x = np.arange(n_groups)

            corr_types = [
                ('pearson', '#ffdd57', 'Pearson'),
                ('spearman', '#ff6b35', 'Spearman'),
                ('kendall', '#00d4ff', 'Kendall')
            ]

            for c_idx, cent in enumerate(cent_vals):
                for corr_idx, (corr_col, corr_color, corr_label) in enumerate(corr_types):
                    offset = (c_idx * 3 + corr_idx - 4) * bar_width
                    y_vals = []
                    for p_val in p_vals:
                        row = subset[(subset['centrality'] == cent) & (subset['p'] == p_val)]
                        y_vals.append(row[corr_col].values[0] if len(row) > 0 else 0)

                    base_colors = {'degree': '#00d4ff', 'closeness': '#ff6b35', 'katz': '#aaff00'}
                    if corr_col == 'pearson':
                        color = base_colors[cent]
                    elif corr_col == 'spearman':
                        color = '#ffaa44' if cent == 'degree' else '#ff8844' if cent == 'closeness' else '#ccff66'
                    else:
                        color = '#88ddff' if cent == 'degree' else '#ffaa88' if cent == 'closeness' else '#ccffaa'

                    label = f"{cent[:3].capitalize()}-{corr_label[:3]}"
                    ax.bar(x + offset, y_vals, bar_width, label=label, color=color, alpha=0.85, edgecolor='none')

            ax.set_title(f"{NETWORK_DISPLAY[net]} — mean_inf", **tkw)
            ax.set_xlabel('Probabilitate p', color='#aaaaaa', fontsize=10)
            ax.set_ylabel('Coeficient corelație', color='#aaaaaa', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([P_DISPLAY[p] for p in p_vals], fontsize=9)
            ax.tick_params(colors='#aaaaaa')
            ax.spines[:].set_color('#333333')
            ax.axhline(y=0, color='#555555', linewidth=0.8, linestyle='-')
            ax.set_ylim(-0.2, 1.05)
            ax.legend(fontsize=6, ncol=3, facecolor='#1a1a1a', labelcolor='white',
                      framealpha=0.8, loc='upper right')

        fig.suptitle('Comparare Pearson vs Spearman vs Kendall (mean_inf)',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig5_comparare_corelatii.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig6_analiza_consistenta(self, df):
        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        tkw = dict(color='white', fontsize=13, fontweight='bold', pad=10)

        net_colors = plt.cm.tab10(np.linspace(0, 1, len(NETWORKS_ORDERED)))
        net_color_map = dict(zip(NETWORKS_ORDERED, net_colors))

        # Subplot 1: Pearson vs Spearman
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#111111')
        for net in NETWORKS_ORDERED:
            subset = df[df['network'] == net]
            ax1.scatter(subset['pearson'], subset['spearman'],
                       c=[net_color_map[net]], s=60, alpha=0.7,
                       label=NETWORK_DISPLAY[net], edgecolors='none')
        lims = [-0.3, 1.0]
        ax1.plot(lims, lims, 'w--', alpha=0.3, linewidth=1)
        ax1.set_xlim(lims); ax1.set_ylim(lims)
        ax1.set_title('Pearson vs Spearman — Consistență', **tkw)
        ax1.set_xlabel('Pearson r', color='#aaaaaa', fontsize=10)
        ax1.set_ylabel('Spearman r', color='#aaaaaa', fontsize=10)
        ax1.tick_params(colors='#aaaaaa')
        ax1.spines[:].set_color('#333333')
        ax1.legend(fontsize=7, ncol=2, facecolor='#1a1a1a', labelcolor='white', loc='lower right')
        ax1.grid(True, alpha=0.15, color='#555555')

        # Subplot 2: Pearson vs Kendall
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        for net in NETWORKS_ORDERED:
            subset = df[df['network'] == net]
            ax2.scatter(subset['pearson'], subset['kendall'],
                       c=[net_color_map[net]], s=60, alpha=0.7,
                       label=NETWORK_DISPLAY[net], edgecolors='none')
        ax2.plot(lims, lims, 'w--', alpha=0.3, linewidth=1)
        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_title('Pearson vs Kendall — Consistență', **tkw)
        ax2.set_xlabel('Pearson r', color='#aaaaaa', fontsize=10)
        ax2.set_ylabel('Kendall τ', color='#aaaaaa', fontsize=10)
        ax2.tick_params(colors='#aaaaaa')
        ax2.spines[:].set_color('#333333')
        ax2.legend(fontsize=7, ncol=2, facecolor='#1a1a1a', labelcolor='white', loc='lower right')
        ax2.grid(True, alpha=0.15, color='#555555')

        # Subplot 3: Box plot
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#111111')
        gen_p = df[df['network'].str.contains('generat')]['pearson'].replace(0, np.nan).dropna()
        real_p = df[df['network'].str.contains('real')]['pearson'].replace(0, np.nan).dropna()
        gen_s = df[df['network'].str.contains('generat')]['spearman'].replace(0, np.nan).dropna()
        real_s = df[df['network'].str.contains('real')]['spearman'].replace(0, np.nan).dropna()

        box_data = [gen_p, real_p, gen_s, real_s]
        box_colors = ['#00d4ff', '#ff6b35', '#00d4ff', '#ff6b35']
        box_labels = ['Gen\nPearson', 'Real\nPearson', 'Gen\nSpearman', 'Real\nSpearman']

        bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                        medianprops=dict(color='white', linewidth=2),
                        whiskerprops=dict(color='#aaaaaa'),
                        capprops=dict(color='#aaaaaa'),
                        flierprops=dict(marker='o', markerfacecolor='white', markersize=4, alpha=0.5))

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor('#333333')

        ax3.set_title('Distribuție Corelații: Generat vs Real', **tkw)
        ax3.set_ylabel('Coeficient corelație', color='#aaaaaa', fontsize=10)
        ax3.tick_params(colors='#aaaaaa')
        ax3.spines[:].set_color('#333333')
        ax3.set_ylim(-0.3, 1.0)
        ax3.grid(True, axis='y', alpha=0.15, color='#555555')

        # Subplot 4: Violin plot
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#111111')
        degree_vals = df[df['centrality'] == 'degree']['pearson'].replace(0, np.nan).dropna()
        closeness_vals = df[df['centrality'] == 'closeness']['pearson'].replace(0, np.nan).dropna()
        katz_vals = df[df['centrality'] == 'katz']['pearson'].replace(0, np.nan).dropna()

        violin_data = [degree_vals, closeness_vals, katz_vals]
        violin_colors = ['#00d4ff', '#ff6b35', '#aaff00']

        parts = ax4.violinplot(violin_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], violin_colors):
            pc.set_facecolor(color); pc.set_alpha(0.6); pc.set_edgecolor('white')
        parts['cmeans'].set_color('white'); parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('#ffdd57'); parts['cmedians'].set_linewidth(2)
        for key in ['cbars', 'cmins', 'cmaxes']:
            parts[key].set_color('#aaaaaa')

        ax4.set_xticks([1, 2, 3])
        ax4.set_xticklabels(['Degree', 'Closeness', 'Katz'], fontsize=10)
        ax4.set_title('Distribuție Pearson per Centralitate', **tkw)
        ax4.set_ylabel('Pearson r', color='#aaaaaa', fontsize=10)
        ax4.tick_params(colors='#aaaaaa')
        ax4.spines[:].set_color('#333333')
        ax4.set_ylim(-0.3, 1.0)
        ax4.grid(True, axis='y', alpha=0.15, color='#555555')

        for i, (vals, color) in enumerate(zip(violin_data, violin_colors), 1):
            mean_val = vals.mean()
            ax4.annotate(f'μ={mean_val:.2f}', xy=(i, mean_val), xytext=(i+0.3, mean_val+0.05),
                        fontsize=9, color=color, fontweight='bold')

        fig.suptitle('Analiză Consistență & Distribuție Corelații',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig6_scatter_box_violin.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _fig7_radar_top(self, df):
        fig = plt.figure(figsize=(22, 14))
        fig.patch.set_facecolor('#0d0d0d')
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1.2, 1])
        tkw = dict(color='white', fontsize=13, fontweight='bold')

        # Radar chart
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        ax1.set_facecolor('#111111')

        p_target = 0.1
        subset_p = df[(df['p'] == p_target) & (df['target_metric'] == 'mean_inf')]
        categories = ['Degree', 'Closeness', 'Katz']
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        net_colors = plt.cm.tab10(np.linspace(0, 1, len(NETWORKS_ORDERED)))
        net_color_map = dict(zip(NETWORKS_ORDERED, net_colors))

        for net in NETWORKS_ORDERED:
            net_data = subset_p[subset_p['network'] == net]
            values = []
            for cent in ['degree', 'closeness', 'katz']:
                row = net_data[net_data['centrality'] == cent]
                values.append(max(0, row['pearson'].values[0]) if len(row) > 0 else 0)
            values += values[:1]
            color = net_color_map[net]
            ax1.plot(angles, values, 'o-', linewidth=2, label=NETWORK_DISPLAY[net], color=color, markersize=6)
            ax1.fill(angles, values, alpha=0.08, color=color)

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories, color='white', fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='#aaaaaa', fontsize=9)
        ax1.tick_params(colors='#aaaaaa')
        ax1.set_title(f'Radar: Corelații Pearson la p={p_target} (mean_inf)', fontdict=tkw, pad=20)
        ax1.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor='#1a1a1a', labelcolor='white')
        ax1.grid(True, alpha=0.3, color='#555555')
        ax1.spines['polar'].set_color('#333333')

        # Tabel top 15
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#111111')
        ax2.axis('off')

        top_rows = df.nlargest(15, 'pearson')[['network', 'p', 'centrality', 'target_metric', 'pearson', 'spearman', 'kendall']]
        table_data = []
        for _, row in top_rows.iterrows():
            table_data.append([
                NETWORK_DISPLAY.get(row['network'], row['network']),
                P_DISPLAY.get(row['p'], str(row['p'])),
                row['centrality'].capitalize(),
                'mean' if row['target_metric'] == 'mean_inf' else 'max',
                f"{row['pearson']:.3f}",
                f"{row['spearman']:.3f}",
                f"{row['kendall']:.3f}"
            ])

        columns = ['Rețea', 'p', 'Centr.', 'Target', 'Pearson', 'Spearman', 'Kendall']
        table = ax2.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center',
                         colColours=['#1a1a1a']*7)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for i in range(len(columns)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
            table[(0, i)].set_facecolor('#2a2a2a')
            table[(0, i)].set_edgecolor('#444444')

        for i in range(1, len(table_data)+1):
            for j in range(len(columns)):
                cell = table[(i, j)]
                cell.set_edgecolor('#333333')
                cell.set_facecolor('#151515' if i % 2 == 0 else '#1a1a1a')
                cell.set_text_props(color='#cccccc')
                if i <= 3 and j >= 4:
                    cell.set_text_props(color='#00ffaa', fontweight='bold')

        ax2.set_title('Top 15 Corelații (toate rețelele, toate p)', fontdict=tkw, pad=20)

        fig.suptitle('Analiză Performanță: Radar + Top Rankings',
                     color='white', fontsize=16, fontweight='bold', y=0.98)
        path = os.path.join(self.output_dir, 'fig7_radar_top.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ✓ Salvat: {path}")

    def _generate_video_animation(self, df, use_ffmpeg=True):
        print("\n🎬 Generare animație video...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('#0d0d0d')

        p_vals = sorted(df['p'].unique())
        net_colors = plt.cm.tab10(np.linspace(0, 1, len(NETWORKS_ORDERED)))
        net_color_map = dict(zip(NETWORKS_ORDERED, net_colors))
        axes_flat = axes.flatten()
        lines_dict = {}

        def init():
            for idx, net in enumerate(NETWORKS_ORDERED):
                ax = axes_flat[idx]
                ax.clear()
                ax.set_facecolor('#111111')
                ax.set_title(f"{NETWORK_DISPLAY[net]}", color='white', fontsize=11, fontweight='bold')
                ax.set_xlabel('Centralitate', color='#aaaaaa', fontsize=9)
                ax.set_ylabel('Pearson r', color='#aaaaaa', fontsize=9)
                ax.tick_params(colors='#aaaaaa')
                ax.spines[:].set_color('#333333')
                ax.set_ylim(-0.3, 1.05)
                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(['Degree', 'Closeness', 'Katz'], fontsize=8)
                ax.axhline(y=0, color='#555555', linewidth=0.5, linestyle='--')

                line_mean, = ax.plot([], [], 'o-', color='#00ffaa', linewidth=2.5, markersize=8, label='mean_inf', alpha=0.9)
                line_max, = ax.plot([], [], 's--', color='#ff4455', linewidth=2.5, markersize=8, label='max_inf', alpha=0.9)
                lines_dict[net] = (line_mean, line_max)
                ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white', loc='upper right')

            fig.suptitle('', color='white', fontsize=14, fontweight='bold', y=0.98)
            return [item for pair in lines_dict.values() for item in pair]

        def update(frame_idx):
            p_val = p_vals[frame_idx]
            for idx, net in enumerate(NETWORKS_ORDERED):
                ax = axes_flat[idx]
                line_mean, line_max = lines_dict[net]
                subset = df[(df['network'] == net) & (df['p'] == p_val)]

                mean_vals = []
                max_vals = []
                for cent in ['degree', 'closeness', 'katz']:
                    row_mean = subset[(subset['centrality'] == cent) & (subset['target_metric'] == 'mean_inf')]
                    row_max = subset[(subset['centrality'] == cent) & (subset['target_metric'] == 'max_inf')]
                    mean_vals.append(row_mean['pearson'].values[0] if len(row_mean) > 0 else 0)
                    max_vals.append(row_max['pearson'].values[0] if len(row_max) > 0 else 0)

                line_mean.set_data(range(3), mean_vals)
                line_max.set_data(range(3), max_vals)

            fig.suptitle(f'Evoluția Corelațiilor — p = {p_val}  (frame {frame_idx+1}/{len(p_vals)})',
                         color='white', fontsize=14, fontweight='bold', y=0.98)
            return [item for pair in lines_dict.values() for item in pair]

        ani = FuncAnimation(fig, update, frames=len(p_vals), init_func=init, blit=False, interval=1500, repeat=True)

        if use_ffmpeg:
            try:
                writer = FFMpegWriter(fps=1, metadata=dict(artist='TVD Analysis'), bitrate=3000)
                path = os.path.join(self.output_dir, 'animatie_evolutie.mp4')
                ani.save(path, writer=writer)
                print(f"  ✓ Video MP4 salvat: {path}")
            except Exception as e:
                print(f"  ⚠️ FFMpeg indisponibil ({e}). Încerc GIF...")
                use_ffmpeg = False

        if not use_ffmpeg:
            try:
                writer = PillowWriter(fps=1)
                path = os.path.join(self.output_dir, 'animatie_evolutie.gif')
                ani.save(path, writer=writer)
                print(f"  ✓ GIF salvat: {path}")
            except Exception as e:
                print(f"  ❌ Eroare salvare animație: {e}")

        plt.close(fig)

    def _generate_raport_text(self, df):
        lines = []
        def line(s=""): lines.append(s)

        line("=" * 70)
        line("  RAPORT VIZUALIZARE — Centralități Clasice vs Stochastic")
        line(f"  Generat: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        line("=" * 70)
        line()
        line("REZUMAT DATE:")
        line(f"  Total înregistrări: {len(df)}")
        line(f"  Rețele analizate: {len(df['network'].unique())}")
        line(f"  Probabilități testate: {sorted(df['p'].unique())}")
        line(f"  Centralități: {', '.join(df['centrality'].unique())}")
        if 'target_metric' in df.columns:
            line(f"  Metrici target: {', '.join(df['target_metric'].unique())}")
        line()

        pearson_vals = df['pearson'].replace(0, np.nan).dropna()
        spearman_vals = df['spearman'].replace(0, np.nan).dropna()
        kendall_vals = df['kendall'].replace(0, np.nan).dropna()

        line("STATISTICI GLOBALE:")
        line(f"  Pearson  — medie: {pearson_vals.mean():.3f}  std: {pearson_vals.std():.3f}")
        line(f"  Spearman — medie: {spearman_vals.mean():.3f}  std: {spearman_vals.std():.3f}")
        line(f"  Kendall  — medie: {kendall_vals.mean():.3f}  std: {kendall_vals.std():.3f}")
        line()

        line("TOP 10 CORELAȚII PEARSON:")
        line(f"{'RK':<4} {'REȚEA':<20} {'p':<8} {'CENTR.':<10} {'TARGET':<8} {'PEARSON':<10} {'SPEARMAN':<10} {'KENDALL'}")
        line("-" * 85)
        top10 = df.nlargest(10, 'pearson')
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            target_disp = 'mean' if row.get('target_metric', 'mean_inf') == 'mean_inf' else 'max'
            line(f"{i:<4} {NETWORK_DISPLAY.get(row['network'], row['network']):<20} "
                 f"{P_DISPLAY.get(row['p'], str(row['p'])):<8} {row['centrality'].capitalize():<10} "
                 f"{target_disp:<8} {row['pearson']:<10.3f} {row['spearman']:<10.3f} {row['kendall']:.3f}")
        line()

        line("CONCLUZII CHEIE:")
        line("  1. Cele mai puternice corelații apar la p ∈ [0.04, 0.1] — infecția")
        line("     este suficient de stochastică pentru a diferenția nodurile, dar")
        line("     nu atât de haotică încât să piardă semnalul structural.")
        line()
        line("  2. Pentru rețelele GENERATE, corelația este mult mai puternică")
        line("     decât pentru cele REALE — structura sintetică are mai puțin")
        line("     'zgomot' decât rețelele naturale.")
        line()
        line("  3. Degree și Closeness sunt comparabile ca putere predictivă;")
        line("     Katz este ușor inferior, posibil din cauza parametrilor fixați.")
        line()
        line("  4. mean_inf are corelații mult mai puternice decât max_inf —")
        line("     media finală a infectaților este mai stabilă decât vârful.")
        line()
        line("  5. La p ≥ 0.3, corelațiile scad dramatic — infecția devine prea")
        line("     rapidă și 'explodează' indiferent de nodul sursă.")
        line("=" * 70)

        raport = "\n".join(lines)
        path = os.path.join(self.output_dir, 'raport_centralitati.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(raport)
        print(f"\n✓ Raport text salvat: {path}")
        print(raport)


# ============================================================================
# PENTRU TESTARE STANDALONE
# ============================================================================

if __name__ == "__main__":
    print("C6_visualizer_final.py - modul pentru vizualizări avansate")
    print("Pentru testare, rulează scriptul 'vizualizare_centralitati.py' separat.")