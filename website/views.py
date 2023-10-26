from django.http import HttpResponse
from django.shortcuts import render
import requests

def index(request):

    if request.method == 'POST':

        api_url = "https://fastapi-production-a6e1.up.railway.app/get-random/"

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

    else:
        quote = ''
        author = ''

    return render(request,'index.html',{
        'quote':quote,
        'author':author,
    })