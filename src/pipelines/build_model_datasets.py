from pprint import pprint

from src.core.model_dataset import build_all_model_datasets


def main() -> None:
    # 공용 베이스, CatBoost wide, TCN sequence 입력을 한 번에 생성한다.
    summary = build_all_model_datasets()
    pprint(summary)


if __name__ == "__main__":
    main()
