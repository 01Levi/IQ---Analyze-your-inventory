from __future__ import annotations

import os
from abc import ABC, abstractmethod

import pandas as pd

SALES_COLUMNS = ["date", "product_id", "product_name", "category",
                 "units_sold", "price", "promotion"]

PRODUCT_COLUMNS = ["product_id", "product_name", "category", "supplier",
                   "unit_cost", "sale_price", "on_hand", "on_order",
                   "reserved", "lead_time_days", "min_order_qty"]


class InventoryDataSource(ABC):

    name: str = "مصدر بيانات"

    @abstractmethod
    def get_sales(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_products(self) -> pd.DataFrame:
        pass

    @staticmethod
    def validate(sales: pd.DataFrame, products: pd.DataFrame) -> None:
        missing_sales = set(SALES_COLUMNS) - set(sales.columns)
        missing_products = set(PRODUCT_COLUMNS) - set(products.columns)
        if missing_sales:
            raise ValueError(f"أعمدة ناقصة في المبيعات: {missing_sales}")
        if missing_products:
            raise ValueError(f"أعمدة ناقصة في المنتجات: {missing_products}")


class DemoCSVDataSource(InventoryDataSource):

    name = "بيانات تجريبية (متجر عطور)"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.sales_path = os.path.join(data_dir, "sample_sales.csv")
        self.products_path = os.path.join(data_dir, "products.csv")

    def _ensure_files(self) -> None:
        if not (os.path.exists(self.sales_path) and os.path.exists(self.products_path)):
            from src.data_generator import save_store_data
            save_store_data(self.data_dir)

    def get_sales(self) -> pd.DataFrame:
        self._ensure_files()
        df = pd.read_csv(self.sales_path, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"])
        return df[SALES_COLUMNS].copy()

    def get_products(self) -> pd.DataFrame:
        self._ensure_files()
        df = pd.read_csv(self.products_path, encoding="utf-8-sig")
        return df[PRODUCT_COLUMNS].copy()


class SallaDataSource(InventoryDataSource):

    name = "متجر سلة (قريباً)"

    def __init__(self, access_token: str | None = None, store_id: str | None = None):
        self.access_token = access_token
        self.store_id = store_id

    def get_sales(self) -> pd.DataFrame:
        raise NotImplementedError(
            "ربط سلة لم يُفعّل بعد. هذه المرحلة الثانية من المشروع."
        )

    def get_products(self) -> pd.DataFrame:
        raise NotImplementedError(
            "ربط سلة لم يُفعّل بعد. هذه المرحلة الثانية من المشروع."
        )


def get_data_source(source: str = "demo", **kwargs) -> InventoryDataSource:
    sources = {"demo": DemoCSVDataSource, "salla": SallaDataSource}
    if source not in sources:
        raise ValueError(f"مصدر غير معروف: {source}. المتاح: {list(sources)}")
    return sources[source](**kwargs)
