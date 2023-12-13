import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm

# Start the browser and open the webpage
driver = webdriver.Chrome()
driver.get("https://www.lutanho.net/play/hex.html")

# WebDriverWait(driver, 3)
time.sleep(3)

RedButton = driver.find_element(By.CSS_SELECTOR, "input[value='Red: Computer']")
RedLevel = driver.find_element(By.CSS_SELECTOR, "input[value='3']")
NewButton = driver.find_element(By.XPATH,
                                "/html/body/div[1]/form/table/tbody/tr/td[5]/table/tbody/tr[9]/td/table/tbody/tr[1]/td/table/tbody/tr/td[1]/input")
MsgLoc = driver.find_element(By.CSS_SELECTOR, "input[name='Msg']")
GetButton = driver.find_element(By.XPATH,
                                "/html/body/div[1]/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[2]/td/table/tbody/tr/td[1]/input")
MoveList = driver.find_element(By.XPATH,
                               "/html/body/div[1]/form/table/tbody/tr/td[5]/table/tbody/tr[11]/td/table/tbody/tr[1]/td/textarea")
MsgFlag = driver.find_element(By.XPATH,
                              "/html/body/div[1]/form/table/tbody/tr/td[5]/table/tbody/tr[9]/td/table/tbody/tr[3]/td/input")

moveList = []
itr_num = 500

def is_ended():
    if "has won !" in MsgLoc.get_attribute("value"):
        return True


def start_new_game():
    NewButton.click()


def get_move_list():
    GetButton.click()
    return MoveList.get_attribute("value")


# 初始化启动（第一次送的）
RedLevel.click()
time.sleep(1)
RedButton.click()

# check if game has ended
while not is_ended():
    time.sleep(1)

move_list = get_move_list()
moveList.append(move_list)

print("Appended 1 time")
time.sleep(3)

# click new game
for i in tqdm(range(itr_num)):
    start_new_game()
    while not is_ended():
        time.sleep(1)
    move_list = get_move_list()
    moveList.append(move_list)
    time.sleep(3)


with open("train_data/moveListTrain.txt", "a") as f:
    for item in moveList:
        f.write("%s\n" % item)

print(f"Successfully done {itr_num} times")
driver.quit()
