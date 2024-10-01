**Интернет-магазин собирает историю покупателей, проводит рассылки предложений и
планирует будущие продажи. Для оптимизации процессов надо выделить пользователей,
которые готовы совершить покупку в ближайшее время**  
---------------------------------------------------------------------------------
**Задачи**    
● Изучить данные    
● Разработать полезные признаки    
● Создать модель для классификации пользователей    
● Улучшить модель и максимизировать метрику roc_auc    
● Выполнить тестирование    
--------------------------------------------------------------------------------
**Цель**    
Предсказать вероятность покупки в течение 90 дней.    
--------------------------------------------------------------------------------
**Разработали модель для предсказаний.** 
![Лучшая модель:](https://github.com/Alexrazz/online_store/blob/master/best_model.png)
Roc_auc на тесте: 0.7402092025961985
--------------------------------------------------------------------------------

![phik:](https://github.com/Alexrazz/online_store/blob/master/corr.png)
![future_importance:](https://github.com/Alexrazz/online_store/blob/master/future_importance.png)
На предсказание прогноза вероятности покупки в течение 90 дней влияют такие признаки как email_subscribe,email_unsubscribe,  
email_hbq_spam, mobile_push_click, reg_client(постоянный клиент/осуществил одну покупку 


