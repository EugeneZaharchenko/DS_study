"""
Приклад реалізації методів HTTP in python: GET, POST, PUT, DELETE:
https://www.tutorialspoint.com/requests/requests_handling_post_put_patch_delete_requests.htm
https://www.w3schools.com/python/module_requests.asp
https://realpython.com/python-requests/
https://www.geeksforgeeks.org/python-requests-tutorial/
https://dev.to/ayabouchiha/sending-get-post-put-delete-requests-in-python-45o8

ДОВІДКОВО:
Відповіді HTTP-сервера
200: запит виконано успішно.
400: запит не сформовано належним чином.
401: несанкціонований запит, клієнт повинен надіслати дані автентифікації.
404: вказаний у запиті ресурс не знайдено.
405 недозволений метод.
500: внутрішня помилка сервера HTTP.
501: запит не реалізований сервером HTTP.
502: служба не доступна.

"""


import requests                        # модуль для роботи з методами HTTP: GET, POST, PUT, DELETE
import json                            # модуль кодування / декодування json формата - JavaScript
# формат.


def GET_test (url: str) -> None:

    '''
    Надсилання запитів GET на сервер з url
    GET : це запит, який використовується для отримання даних або інформації з вказаного сервера.
    :param url: http адреса серверу
    :return:    аж нічого
    '''

    response = requests.get(url)        # ініціалізація доступу до сервера
    print(response.status_code)         # перевірка успішності доступу до сервера, успішна
    # відповідь для get 200
    print(response.text)                # відображення вмісту сторінки за url
    print('url_GET', response.url)      # url

    return


def POST_test (url: str) -> None:

    """
    Надсилання запитів POST на сервер з url
    POST : це запит, який використовується для надсилання інформації або даних на певний сервер.
    :param url: http адреса сервера
    :return:    аж нічого
    """

    # інформаціі, що надається до сервера
    data = {'title': 'тестування_POST_для_заняття', 'ОК': '_тестування_POST_успішне', 'userId': 5, }
    headers = {'content-type': 'application/json; charset=UTF-8'}
    response = requests.post(url, data=json.dumps(data), headers=headers)   # ініціалізація
    # надfння інформаціі до серверу
    print(response.status_code)        # перевірка успішності доступу до сервера, успішна
    # відповідь для post 201
    print(response.text)               # відображення вмісту сторінки за url
    print('url_POST', response.url)    # url

    return


def PUT_test (url: str) -> None:

    '''
    Надсилання запитів PUT на сервер з url
    PUT : це запит, який використовується для створення або оновлення ресурсу на певному сервері.
    :param url: http адреса сервера
    :return:    аж нічого
    '''

    # інформаціі, що надається до сервера
    data = {'id': 1, 'userId': 2, 'title': 'тестування_PUT_для_заняття', 'body': 'тестування_PUT_успішне'}
    headers = {'Content-Type': 'application/json; charset=UTF-8'}

    # ініціалізація надпння інформаціі до серверу
    response = requests.put(url, data=json.dumps(data), headers=headers)
    print(response.status_code)       # перевірка успішності доступу до серверу, успішна відповідь для post 200
    print(response.text)              # відображення вмісту сторінки за url
    print('url_PUT', response.url)    # url

    return


def DELETE_test (url: str) -> None:

    '''
    Надсилання запитів DELETE на сервер з url
    DELETE : це запит, який використовується для видалення певного ресурсу на сервері.
    :param url: http адреса серверу
    :return:    аж нічого
    '''

    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    # ініціалізація видалення інформації з серверу
    response = requests.delete(url, headers=headers)
    print(response.status_code)   # перевірка успішності доступу до серверу, успішна відповідь для post 200
    print('url_DELETE', response.url)  # url

    return


if __name__ == '__main__':

    print('Оберіть напрям досліджень:')
    print('1 - GET:    отримання даних з сервера')
    print('2 - POST:   надсилання  даних на сервер')
    print('3 - PUT:    створення або оновлення ресурсу на сервері')
    print('4 - DELETE: видалення певного ресурсу на сервері')
    mode = int(input('mode:'))

    # url = 'https://www.rbc.ua'
    url = 'https://jsonplaceholder.typicode.com/posts/1'
    print(url)


    if (mode == 1):
        print('Обрано: GET')
        print('-----------------------------------------------------------------------------')
        GET_test(url)   # отримання інформації з сервера.
        print('-----------------------------------------------------------------------------')


    if (mode == 2):
        print('Обрано: POST')
        print('-----------------------------------------------------------------------------')
        POST_test(url)  # надсилання інформації на сервер
        print('-----------------------------------------------------------------------------')


    if (mode == 3):
        print('Обрано: PUT')
        print('-----------------------------------------------------------------------------')
        PUT_test(url)  # спроба створення або оновлення ресурсу на певному сервері
        print('-----------------------------------------------------------------------------')


    if (mode == 4):
        print('Обрано: DELETE')
        print('-----------------------------------------------------------------------------')
        DELETE_test(url)  # спроба видалити інформацію на сервері
        print('-----------------------------------------------------------------------------')


