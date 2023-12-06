from django.http import HttpResponse
from django.shortcuts import render
import requests

def index(request):

    quote = ''
    author = ''

    if request.method == 'POST':

        if 'random_quote' in request.POST:

            api_url = "https://fastapi-production-6e77.up.railway.app/get-random/"

            try:
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    quote = data["quote"]
                    author = data['author']
                else:
                    return HttpResponse(f"Request failed with status code: {response.status_code}",
                                        content_type="text/plain", status=response.status_code)

            except requests.exceptions.RequestException as e:
                return HttpResponse(f"An error occurred: {e}", content_type="text/plain", status=500)

        elif 'search_quote' in request.POST:

            search_query = request.POST['your_input_name']

            api_url = f"https://fastapi-production-6e77.up.railway.app/get-by-author/?author={search_query}"

            try:
                response = requests.get(api_url)

                if response.status_code == 200:
                    data = response.json()
                    quote = data["quote"]
                    author = data['author']
                else:
                    return HttpResponse(f"Request failed with status code: {response.status_code}",
                                        content_type="text/plain", status=response.status_code)

            except requests.exceptions.RequestException as e:
                return HttpResponse(f"An error occurred: {e}", content_type="text/plain", status=500)

    return render(request,'index.html',{
        'quote':quote,
        'author':author,
    })
