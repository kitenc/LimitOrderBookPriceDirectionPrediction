import pandas as pd


#_____________________________________________________________________________
#
# Load Message File
#_____________________________________________________________________________
#		
# 		StartTime and EndTime give the theoretical beginning 
# 		and end time of the output file in milliseconds after 		
# 		mid night. LEVEL refers to the number of levels of the 
# 		requested limit order book.
# 	Columns:
# 	    1.) Time: 		
# 				Seconds after midnight with decimal 
# 				precision of at least milliseconds 
# 				and up to nanoseconds depending on 
# 				the requested period
# 	    2.) Type:
# 				1: Submission of a new limit order
# 				2: Cancellation (Partial deletion 
# 				   of a limit order)
# 				3: Deletion (Total deletion of a limit order)
# 				4: Execution of a visible limit order			   	 
# 				5: Execution of a hidden limit order
# 				7: Trading halt indicator 				   
# 				   (Detailed information below)
# 	    3.) Order ID: 	
# 				Unique order reference number 
# 				(Assigned in order flow)
# 	    4.) Size: 		
# 				Number of shares
# 	    5.) price: 		
# 				Dollar price times 10000 
# 				(i.e., A stock price of $91.14 is given 
# 				by 911400)
# 	    6.) Direction:
# 				-1: Sell limit order
# 				1: Buy limit order
				
# 				Note: 
# 				Execution of a sell (buy) limit
# 				order corresponds to a buyer (seller) 
# 				initiated trade, i.e. Buy (Sell) trade.
#
#		When trading halts, a message of type '7' is written into the 
#		'message' file. The corresponding price and trade direction 
#		are set to '-1' and all other properties are set to '0'. 
#		Should the resume of quoting be indicated by an additional 
#		message in NASDAQ's Historical TotalView-ITCH files, another 
#		message of type '7' with price '0' is added to the 'message' 
#		file. Again, the trade direction is set to '-1' and all other 
#		fields are set to '0'. 
#		When trading resumes a message of type '7' and 
#		price '1' (Trade direction '-1' and all other 
#		entries '0') is written to the 'message' file. For messages 
#		of type '7', the corresponding order book rows contain a 
#		duplication of the preceding order book state. The reason 
#		for the trading halt is not included in the output.
#						
#		Example: Stylized trading halt messages in 'message' file.				
#	
#		Halt: 			36023	| 7 | 0 | 0 | -1 | -1
#											...
#		Quoting: 		36323 	| 7 | 0 | 0 | 0  | -1
#											...
#		Resume Trading:		36723   | 7 | 0 | 0 | 1  | -1
#											...
#		The vertical bars indicate the different columns in the  
#		message file.
# ----------------------------------------------------------
# 
# 


def read_messagebook_data(ticker, year, month, day):
    theMessageBookFileName =  ticker+"_" + year + "-" + month + '-' + day + "_34200000_57600000_message_10.csv"
    path = "./data/" + theMessageBookFileName
    theMessageBook = pd.read_csv(path, names = ['Time','Type','OrderID','Size','Price','TradeDirection'])
    theMessageBook['ticker'] = ticker

    startTrad = 9.5*60*60       # 9:30:00.000 in ms after midnight
    endTrad = 16*60*60        # 16:00:00.000 in ms after midnight

    theMessageBookFiltered = theMessageBook[theMessageBook['Time'] >= startTrad] 
    theMessageBookFiltered = theMessageBookFiltered[theMessageBookFiltered['Time'] <= endTrad]

    return theMessageBookFiltered


#_____________________________________________________________________________
#
# Load Order Book File
#_____________________________________________________________________________

# Columns:
#     1.) Ask Price 1: 	Level 1 Ask Price 	(Best Ask)
#     2.) Ask Size 1: 	Level 1 Ask Volume 	(Best Ask Volume)
#     3.) Bid price 1: 	Level 1 Bid Price 	(Best Bid)
#     4.) Bid Size 1: 	Level 1 Bid Volume 	(Best Bid Volume)
#     5.) Ask Price 2: 	Level 2 Ask Price 	(2nd Best Ask)
#     ...

# Notes: 	 
# ------

#     - Levels:
    
#     The term level refers to occupied price levels. This implies 
#     that the difference between two levels in the LOBSTER output 
#     is not necessarily the minimum ticks size.

#     - Unoccupied Price Levels:

#     When the selected number of levels exceeds the number of levels 
#     available the empty order book positions are filled with dummy 
#     information to guarantee a symmetric output. The extra bid 
#     and/or ask prices are set to -9999999999 and 9999999999, 
#     respectively. The Corresponding volumes are set to 0. 



# Load data
def read_orderbook_data(ticker, nlevels, year, month, day):
    theOrderBookFileName =  ticker+"_" + year + "-" + month + '-' + day + "_34200000_57600000_orderbook_10.csv"
    path = './data/' + theOrderBookFileName
    col = ['Ask_Price','Ask_Size','Bid_Price','Bid_Size']

    theNames = []
    for i in range(1, nlevels + 1):
        for j in col:
            theNames.append(str(j) + '_' + str(i))

    theOrderBook = pd.read_csv(path, names = theNames)


    return theOrderBook
    