from typing import Union

from fastapi import FastAPI
from fastapi import UploadFile, File

import pandas as pd
import requests
import io
import calendar

#ultimos que agregue
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
app = FastAPI()



# app = FastAPI()
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

df = None

meses_espanol = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


from fastapi.responses import JSONResponse

from fastapi.responses import JSONResponse

from typing import Optional

@app.get('/cantidad_filmaciones_mes', response_class=HTMLResponse)
def cantidad_filmaciones_mes(request: Request, mes: str, formato: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        mes_lower = mes.lower()
        if mes_lower in meses_espanol:
            mes_num = meses_espanol[mes_lower]
            mes_str = calendar.month_name[mes_num]
            cantidad = df[df['release_month'] == mes_num].shape[0]

            if formato == 'json':
                return JSONResponse({"mes_str": mes_str, "cantidad": cantidad})
            else:
                return templates.TemplateResponse("result.html", {"request": request, "mes_str": mes_str, "cantidad": cantidad})
        else:
            mensaje_error = "Mes inválido. Ingrese un mes válido en idioma español."

            if formato == 'json':
                return JSONResponse({"error": mensaje_error})
            else:
                return templates.TemplateResponse("error.html", {"request": request, "message": mensaje_error})
    else:
        mensaje_error = "No se ha cargado ningún archivo CSV."

        if formato == 'json':
            return JSONResponse({"error": mensaje_error})
        else:
            return templates.TemplateResponse("error.html", {"request": request, "message": mensaje_error})


@app.post("/cargar_csv")
def cargar_csv(file: UploadFile = File(...)):
    global df  # Acceder a la variable global df

    try:
        df = pd.read_csv(file.file, sep=';')
        return {"message": "Archivo CSV cargado correctamente"}
    except pd.errors.ParserError:
        return {"error": "Error al analizar el archivo CSV"}
    

dias_espanol = {
    'lunes': 1,
    'martes': 2,
    'miércoles': 3,
    'jueves': 4,
    'viernes': 5,
    'sábado': 6,
    'domingo': 7
}

@app.get('/cantidad_filmaciones_dia', response_class=HTMLResponse)
def cantidad_filmaciones_dia(request: Request, dia: str, formato: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        dia_lower = dia.lower()
        if dia_lower in dias_espanol:
            dia_num = dias_espanol[dia_lower]
            cantidad = df[df['release_day'] == dia_num].shape[0]

            if formato == 'json':
                return JSONResponse({"cantidad": cantidad})
            else:
                return templates.TemplateResponse("result_dia.html", {"request": request, "dia": dia, "cantidad": cantidad})
        else:
            mensaje_error = "Día inválido. Ingrese un día válido en idioma español."

            if formato == 'json':
                return JSONResponse({"error": mensaje_error})
            else:
                return templates.TemplateResponse("error_dia.html", {"request": request, "message": mensaje_error})
    else:
        mensaje_error = "No se ha cargado ningún archivo CSV."

        if formato == 'json':
            return JSONResponse({"error": mensaje_error})
        else:
            return templates.TemplateResponse("error_dia.html", {"request": request, "message": mensaje_error})

from fastapi.responses import JSONResponse

from typing import Optional

from fastapi.responses import HTMLResponse, JSONResponse

@app.get('/score_titulo')
def score_titulo(request: Request, titulo: str, format: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        pelicula = df[df['title'] == titulo]

        if not pelicula.empty:
            año_estreno = pelicula['release_year'].iloc[0]
            score = pelicula['popularity'].iloc[0]

            if format.lower() == 'json':
                print("La respuesta se envió en formato JSON.")
                # Convertir int64 a int
                año_estreno = int(año_estreno)
                score = float(score)
                return JSONResponse({
                    "titulo": titulo,
                    "año_estreno": año_estreno,
                    "score": score
                })
            else:
                return templates.TemplateResponse("result_score.html", {"request": request, "titulo": titulo, "año_estreno": año_estreno, "score": score})
        else:
            return templates.TemplateResponse("error_score.html", {"request": request, "message": "No se encontró la película especificada"})
    else:
        return templates.TemplateResponse("error_score.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})


# Ruta para obtener la cantidad de votos y promedio de un título
@app.get('/votos_titulo', response_class=HTMLResponse)
async def votos_titulo(
    request: Request,
    titulo: str,
    formato: Optional[str] = 'html'
):
    global df  # Acceder a la variable global df

    if df is not None:
        pelicula = df[df['title'] == titulo]

        if not pelicula.empty:
            votos = pelicula['popularity'].values[0]
            promedio = pelicula['vote_average'].values[0]

            if votos >= 2000:
                if formato == 'json':
                    return JSONResponse({"titulo": titulo, "votos": votos, "promedio": promedio})
                else:
                    return templates.TemplateResponse("result_votos.html", {"request": request, "titulo": titulo, "votos": votos, "promedio": promedio})
            else:
                if formato == 'json':
                    return JSONResponse({"error": "La película no cumple con la condición de tener al menos 2000 valoraciones."})
                else:
                    return templates.TemplateResponse("error_votos.html", {"request": request, "message": "La película no cumple con la condición de tener al menos 2000 valoraciones."})
        else:
            if formato == 'json':
                return JSONResponse({"error": "No se encontró la filmación especificada."})
            else:
                return templates.TemplateResponse("error_votos.html", {"request": request, "message": "No se encontró la filmación especificada."})
    else:
        if formato == 'json':
            return JSONResponse({"error": "No se ha cargado ningún archivo CSV"})
        else:
            return templates.TemplateResponse("error_votos.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})


##ultimas 2 funciones
from typing import Optional
from fastapi.responses import JSONResponse

@app.get('/get_actor')
def get_actor(request: Request, nombre_actor: str, formato: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        actor_films = df[df['casting'].apply(lambda x: nombre_actor in str(x) if isinstance(x, str) else False)]

        if not actor_films.empty:
            cantidad_films = actor_films.shape[0]
            retorno_promedio = actor_films['return'].mean()

            if formato == 'json':
                return JSONResponse({"nombre_actor": nombre_actor, "cantidad_films": cantidad_films, "retorno_promedio": retorno_promedio})
            else:
                return templates.TemplateResponse("result_actor.html", {"request": request, "nombre_actor": nombre_actor, "cantidad_films": cantidad_films, "retorno_promedio": retorno_promedio})
        else:
            if formato == 'json':
                return JSONResponse({"error": "No se encontró al actor especificado"})
            else:
                return templates.TemplateResponse("error_actor.html", {"request": request, "message": "No se encontró al actor especificado"})
    else:
        if formato == 'json':
            return JSONResponse({"error": "No se ha cargado ningún archivo CSV"})
        else:
            return templates.TemplateResponse("error_actor.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})


from typing import Optional
from fastapi.responses import JSONResponse

@app.get('/get_director')
def get_director(request: Request, nombre_director: str, formato: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        director_films = df[df['director'].str.contains(nombre_director, case=False, na=False)]

        if not director_films.empty:
            director_success = director_films['return'].sum()
            films_data = []

            for _, row in director_films.iterrows():
                film_title = row['title']
                release_date = row['release_date']
                film_return = row['return']
                film_budget = row['budget']
                film_revenue = row['revenue']
                films_data.append({'title': film_title, 'release_date': release_date, 'return': film_return,
                                   'budget': film_budget, 'revenue': film_revenue})

            if formato == 'json':
                return JSONResponse({"nombre_director": nombre_director, "director_success": director_success, "films_data": films_data})
            else:
                return templates.TemplateResponse("result_director.html", {"request": request, "nombre_director": nombre_director, "director_success": director_success, "films_data": films_data})
        else:
            if formato == 'json':
                return JSONResponse({"error": "No se encontró al director especificado"})
            else:
                return templates.TemplateResponse("error_director.html", {"request": request, "message": "No se encontró al director especificado"})
    else:
        if formato == 'json':
            return JSONResponse({"error": "No se ha cargado ningún archivo CSV"})
        else:
            return templates.TemplateResponse("error_director.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})


#modelo de ml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def get_recommendations(title, df):
    # Define a TF-IDF Vectorizer Object. Remove all English stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    df['overview'] = df['overview'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df['title'].iloc[movie_indices]


# @app.get('/recommend', response_class=HTMLResponse)
# def recommend_movies(request: Request, title: str):
#     global df  # Acceder a la variable global df

#     if df is not None:
#         if title in df['title'].values:
#             recommendations = get_recommendations(title, df)
#             return templates.TemplateResponse("result_recommend.html", {"request": request, "title": title, "recommendations": recommendations.tolist()})
#         else:
#             return templates.TemplateResponse("error_recommend.html", {"request": request, "message": "No se encontró la película especificada"})
#     else:
#         return templates.TemplateResponse("error_recommend.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})

from typing import Optional
from fastapi.responses import JSONResponse

@app.get('/recommend')
def recommend_movies(request: Request, title: str, formato: Optional[str] = 'html'):
    global df  # Acceder a la variable global df

    if df is not None:
        if title in df['title'].values:
            recommendations = get_recommendations(title, df)

            if formato == 'json':
                return JSONResponse({"title": title, "recommendations": recommendations.tolist()})
            else:
                return templates.TemplateResponse("result_recommend.html", {"request": request, "title": title, "recommendations": recommendations})
        else:
            if formato == 'json':
                return JSONResponse({"error": "No se encontró la película especificada"})
            else:
                return templates.TemplateResponse("error_recommend.html", {"request": request, "message": "No se encontró la película especificada"})
    else:
        if formato == 'json':
            return JSONResponse({"error": "No se ha cargado ningún archivo CSV"})
        else:
            return templates.TemplateResponse("error_recommend.html", {"request": request, "message": "No se ha cargado ningún archivo CSV"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)