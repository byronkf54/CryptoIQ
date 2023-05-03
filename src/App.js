import React, { useEffect, useState } from 'react';
import './CSS/App.css';
import axios from 'axios'
import Coin from './Coin';
import Navbar from "./Navbar";


function App() {
  const [coins,setCoins] = useState([])
  const [ratings, setRatings] = useState({});
  const [search,setSearch] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      const result = await axios(
        'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false'
      );
      setCoins(result.data);
    };

    const fetchRatings = async () => {
        try {
            const response = await fetch('/ratings.json');
            const data = await response.json();
            setRatings(data);
        } catch(error) {
          console.log(error)
        }
    }

    fetchData().then(() => fetchRatings());

  }, []);


  const handleChange = e =>{
    setSearch(e.target.value)
  }


  const filteredCoins = coins.filter(coin=>
    coin.name.toLowerCase().includes(search.toLowerCase())
    )

  return (
    <div className="coin-app">
      <Navbar handleChange={handleChange} />
      <table>
        <thead>
          <tr>
            <th>Coin</th>
            <th>Price</th>
            <th>Volume</th>
            <th>Day Price Change</th>
            <th>Market Cap</th>
            <th>Rating</th>
          </tr>
        </thead>
        <tbody>
          {filteredCoins.map(coin=>{
            const rating = ratings[coin.symbol]; // retrieve the rating for the current coin
            console.log(rating)
            return(
              <Coin
                key={coin.id}
                name={coin.name}
                image={coin.image}
                symbol={coin.symbol}
                marketcap={coin.market_cap}
                volume={coin.total_volume}
                price={coin.current_price}
                pricechange={coin.price_change_percentage_24h}
                rating={rating}
              />
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default App;
