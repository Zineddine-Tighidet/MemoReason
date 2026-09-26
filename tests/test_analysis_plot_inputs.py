"""Figure rendering needs computed tables, not an archived result release."""

import csv

from scripts.analysis import plots


def test_render_all_uses_only_supplied_tables(tmp_path, monkeypatch):
    table_dir = tmp_path / "partial"
    table_dir.mkdir()
    percentages = (0, 10, 20, 30, 50, 80, 90, 100)
    accuracies = (80, 78, 73, 70, 68, 65, 64, 60)
    for model, _ in plots.PARTIAL_MODELS:
        with (table_dir / f"{model}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("replaced_percent", "accuracy_percent", "count_total"))
            writer.writerows((percentage, accuracy, 1200 if percentage == 0 else 12000)
                             for percentage, accuracy in zip(percentages, accuracies, strict=True))

    rendered = {}

    def capture_figure(figure, output, stem):
        assert output == tmp_path
        rendered[stem] = tuple(figure.axes[0].lines[1].get_ydata())
        plots.plt.close(figure)

    monkeypatch.setattr(plots, "save", capture_figure)
    report = plots.render_all(tmp_path)

    assert rendered["partial_accuracy_curves"] == accuracies
    assert set(rendered) == {"partial_accuracy_curves", "partial_accuracy_qwen27"}
    assert report["partial_curves"]["sources"] == {
        model: f"partial/{model}.csv" for model, _ in plots.PARTIAL_MODELS
    }
    assert set(tmp_path.iterdir()) == {table_dir}
