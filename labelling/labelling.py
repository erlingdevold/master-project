from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base
import dca as dca
import xarray as xr

engine = create_engine('postgresql://postgres:example@localhost:5420/postgres', echo=True)

def create_select_query(items : list,limit = 0):
    query = "SELECT "
    for item in items:
        if item == "*":
            query = "SELECT *  "
        else:
            query += '"'+ item +'"'+ ", "
    query = query[:-2]
    query += " FROM dca" 
    query += f" LIMIT {limit}" if limit else ""
    return query

def get_columns():
    with engine.connect() as conn:
        res = conn.execute(text("select * from dca limit 1;"))  
    return res.keys()

def read_data():
    
    with engine.connect() as conn:
        res = conn.execute(text(create_select_query(["*"])))  
    return res

def read_data_to_xarray(data):
    
    columns = list(data.keys())

    xr_data = xr.DataArray(data.fetchall())

    xr_data = xr_data.assign_coords({"dim_1": columns})
    xr_data = xr_data.to_dataset(dim="dim_1")

    return xr_data

def write_to_file(data):
    data.to_netcdf("dca_labels.nc")


def read_file(file):
    data = xr.open_dataset(file)
    return data

if __name__ == "__main__":
    data = read_data()

    labels = read_data_to_xarray(data)
    # labels.drop()

    write_to_file(labels)

    labels =read_file("dca_labels.nc")
    print(labels)


    # print(res.all())

    # for row in res:
    #     print(row)

# session = sessionmaker(bind=engine)

# s = session()

# res = s.query(dca.DCA).limit(1)
# print(res.all())




